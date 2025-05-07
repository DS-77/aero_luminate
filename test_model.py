import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import time
import gc
from models import adapter_shadow, shadow_diff
import json
from typing import Union, List, Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Shadow detection and removal testing script')

    # Model weights paths
    parser.add_argument('--adapter_weights', required=True, type=str,
                        help='Path to the adapter_shadow model weights')
    parser.add_argument('--diffusion_weights', required=True, type=str,
                        help='Path to the shadow_diffusion model weights')
    parser.add_argument('--diffusion_config', required=True, type=str,
                        help='Path to the shadow_diffusion model config JSON')
    parser.add_argument('--sam_checkpoint', required=True, type=str,
                        help='Path to the SAM model checkpoint')

    # Input options
    parser.add_argument('--input', required=True, type=str,
                        help='Path to input image or directory of images')
    parser.add_argument('--output_dir', required=True, type=str,
                        help='Directory to save output results')

    # Additional options
    parser.add_argument('--device', default='cuda', type=str,
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--image_size', default=256, type=int,
                        help='Size to resize images for processing')
    parser.add_argument('--save_intermediates', action='store_true',
                        help='Save intermediate shadow masks')
    parser.add_argument('--time_benchmark', action='store_true',
                        help='Benchmark and report processing times')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size for directory processing (default: 1)')
    parser.add_argument('--viz_results', action='store_true',
                        help='Visualize results with matplotlib')

    return parser.parse_args()


class ShadowDetectionRemoval:
    def __init__(
            self,
            adapter_weights: str,
            diffusion_weights: str,
            diffusion_config: str,
            sam_checkpoint: str,
            device: str = 'cuda',
            image_size: int = 256
    ):
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        # Load adapter_shadow model
        print("Loading adapter_shadow model...")
        self.adapter_shadow = self.load_adapter_shadow_model(adapter_weights, sam_checkpoint)

        # Load shadow_diffusion model
        print("Loading shadow_diffusion model...")
        self.shadow_diff = self.load_shadow_diffusion_model(diffusion_weights, diffusion_config)

        print(f"Models loaded successfully on {device}")

    def load_adapter_shadow_model(self, weights_path: str, sam_checkpoint: str) -> torch.nn.Module:
        adapter_args = self.create_adapter_args()
        model = adapter_shadow.adapt_sam_model_registry('vit_b', None, checkpoint=sam_checkpoint)
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location=self.device))
        else:
            raise FileNotFoundError(f"Adapter shadow weights not found at {weights_path}")

        model = model.to(self.device)
        model.eval()

        return model

    def create_adapter_args(self):
        adapter_args = argparse.Namespace()
        adapter_args.vpt = True
        adapter_args.all = True
        adapter_args.net = "sam"
        adapter_args.lora = True
        adapter_args.backbone = "b5"
        adapter_args.ratio_bt = 1.0
        adapter_args.mb_ratio = 0.25
        adapter_args.small_size = True
        adapter_args.multi_branch = True
        adapter_args.skip_adapter = True
        adapter_args.plug_image_adapter = True
        return adapter_args

    def load_shadow_diffusion_model(self, weights_path: str, config_path: str) -> torch.nn.Module:
        with open(config_path, 'r') as f:
            shadow_diff_opt = json.load(f)

        # Create model
        model = shadow_diff.create_model(shadow_diff_opt)
        model.set_new_noise_schedule(
            shadow_diff_opt['model']['beta_schedule']['train'],
            schedule_phase='train'
        )

        if os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            raise FileNotFoundError(f"Shadow diffusion weights not found at {weights_path}")

        model = model.to(self.device)
        model.eval()

        return model

    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, Image.Image]:
        orig_img = Image.open(image_path).convert('RGB')
        self.orig_size = orig_img.size
        img_tensor = self.transform(orig_img)
        img_tensor = img_tensor.unsqueeze(0)

        return img_tensor, orig_img

    @torch.no_grad()
    def detect_shadow(self, img_tensor: torch.Tensor) -> torch.Tensor:
        img_tensor = img_tensor.to(self.device)
        shadow_mask = self.adapter_shadow(img_tensor)
        shadow_mask = torch.sigmoid(shadow_mask)

        return shadow_mask

    @torch.no_grad()
    def remove_shadow(self, img_tensor: torch.Tensor, shadow_mask: torch.Tensor) -> torch.Tensor:
        img_tensor = img_tensor.to(self.device)
        shadow_mask = shadow_mask.to(self.device)
        shadow_free_img = self.shadow_diff(shadow_mask, img_tensor)

        return shadow_free_img

    def postprocess_image(self, img_tensor: torch.Tensor) -> Image.Image:
        img_tensor = torch.clamp(img_tensor, 0, 1)
        img_np = img_tensor.squeeze(0).cpu().numpy()
        img_np = np.transpose(img_np, (1, 2, 0))
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
        img_pil = img_pil.resize(self.orig_size, Image.BICUBIC)

        return img_pil

    def postprocess_mask(self, mask_tensor: torch.Tensor) -> Image.Image:
        mask_tensor = torch.clamp(mask_tensor, 0, 1)
        mask_np = mask_tensor.squeeze(0).squeeze(0).cpu().numpy()
        mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
        mask_pil = mask_pil.resize(self.orig_size, Image.BICUBIC)

        return mask_pil

    def process_single_image(
            self,
            image_path: str,
            output_dir: str,
            save_intermediates: bool = False,
            time_benchmark: bool = False,
            visualize: bool = False
    ) -> dict:
        times = {"preprocessing": 0, "shadow_detection": 0, "shadow_removal": 0, "postprocessing": 0}
        if time_benchmark:
            start = time.time()

        img_tensor, orig_img = self.preprocess_image(image_path)

        if time_benchmark:
            times["preprocessing"] = time.time() - start
            start = time.time()

        shadow_mask = self.detect_shadow(img_tensor)

        if time_benchmark:
            times["shadow_detection"] = time.time() - start
            start = time.time()

        shadow_free_img = self.remove_shadow(img_tensor, shadow_mask)

        if time_benchmark:
            times["shadow_removal"] = time.time() - start
            start = time.time()

        shadow_free_pil = self.postprocess_image(shadow_free_img)
        filename = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{filename}_shadowfree.png")
        shadow_free_pil.save(output_path)

        if save_intermediates:
            mask_pil = self.postprocess_mask(shadow_mask)
            mask_path = os.path.join(output_dir, f"{filename}_shadowmask.png")
            mask_pil.save(mask_path)

        if time_benchmark:
            times["postprocessing"] = time.time() - start

        if visualize:
            mask_pil = self.postprocess_mask(shadow_mask)
            self.visualize_results(orig_img, mask_pil, shadow_free_pil, filename, output_dir)

        return {"path": output_path, "times": times}

    def process_directory(
            self,
            input_dir: str,
            output_dir: str,
            batch_size: int = 1,
            save_intermediates: bool = False,
            time_benchmark: bool = False,
            visualize: bool = False
    ) -> List[str]:
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']
        image_files = [
            os.path.join(input_dir, f) for f in os.listdir(input_dir)
            if os.path.splitext(f.lower())[1] in valid_extensions
        ]

        results = []
        total_times = {"preprocessing": 0, "shadow_detection": 0, "shadow_removal": 0, "postprocessing": 0}

        print(f"Processing {len(image_files)} images...")

        for img_path in tqdm(image_files):
            try:
                result = self.process_single_image(
                    img_path,
                    output_dir,
                    save_intermediates,
                    time_benchmark,
                    visualize
                )
                results.append(result["path"])
                if time_benchmark:
                    for key in total_times:
                        total_times[key] += result["times"][key]
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                    gc.collect()

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        if time_benchmark and len(image_files) > 0:
            print("\nAverage processing times:")
            for key, value in total_times.items():
                avg_time = value / len(image_files)
                print(f"  {key}: {avg_time:.4f} seconds")

            total_avg = sum(total_times.values()) / len(image_files)
            print(f"  Total: {total_avg:.4f} seconds per image")

        return results

    def visualize_results(
            self,
            original_img: Image.Image,
            shadow_mask: Image.Image,
            shadow_free: Image.Image,
            filename: str,
            output_dir: str
    ):
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(original_img)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(shadow_mask, cmap='gray')
        plt.title('Shadow Mask')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(shadow_free)
        plt.title('Shadow-Free Image')
        plt.axis('off')

        plt.tight_layout()

        # Save visualisation
        viz_path = os.path.join(output_dir, f"{filename}_visualization.png")
        plt.savefig(viz_path)
        plt.close()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    shadow_pipeline = ShadowDetectionRemoval(
        adapter_weights=args.adapter_weights,
        diffusion_weights=args.diffusion_weights,
        diffusion_config=args.diffusion_config,
        sam_checkpoint=args.sam_checkpoint,
        device=args.device,
        image_size=args.image_size
    )

    if os.path.isfile(args.input):
        print(f"Processing single image: {args.input}")
        result = shadow_pipeline.process_single_image(
            args.input,
            args.output_dir,
            args.save_intermediates,
            args.time_benchmark,
            args.viz_results
        )
        print(f"Shadow-free image saved to: {result['path']}")

        if args.time_benchmark:
            print("\nProcessing times:")
            for key, value in result["times"].items():
                print(f"  {key}: {value:.4f} seconds")
            print(f"  Total: {sum(result['times'].values()):.4f} seconds")

    elif os.path.isdir(args.input):
        print(f"Processing directory: {args.input}")
        results = shadow_pipeline.process_directory(
            args.input,
            args.output_dir,
            args.batch_size,
            args.save_intermediates,
            args.time_benchmark,
            args.viz_results
        )
        print(f"Processed {len(results)} images. Results saved to: {args.output_dir}")
    else:
        print(f"Error: Input path {args.input} does not exist")


if __name__ == "__main__":
    main()