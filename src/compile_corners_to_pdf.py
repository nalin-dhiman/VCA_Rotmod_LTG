"""
Compile all PNGs in a directory into a single PDF.
"""
import os
import glob
from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

def compile_images_to_pdf(image_dir, output_pdf, title_suffix=""):
    images = sorted(glob.glob(os.path.join(image_dir, '*.png')))
    if not images:
        print(f"No images found in {image_dir}")
        return

    print(f"Compiling {len(images)} images to {output_pdf}...")
    
    with PdfPages(output_pdf) as pdf:
        for img_path in images:
            try:
                img = Image.open(img_path)
                # Create a figure to hold the image
                # Assuming standard aspect ratio, let's fit to letter page
                fig = plt.figure(figsize=(8.5, 11))
                ax = fig.add_subplot(111)
                ax.imshow(img)
                ax.axis('off')
                
                # Title from filename
                name = os.path.basename(img_path).split('_')[0]
                ax.set_title(f"{name} {title_suffix}", fontsize=12)
                
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                print(f"Failed to add {img_path}: {e}")
                
    print("PDF compilation complete.")

if __name__ == "__main__":
    compile_images_to_pdf(
        "results_mcmc_refined/corner_plots",
        "results_mcmc_refined/all_corner_plots.pdf",
        title_suffix="- Corner Plot"
    )
