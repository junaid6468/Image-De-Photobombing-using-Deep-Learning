import subprocess
def inpaint_image():
    try:
        input_dir = "dataset"
        output_dir = "output_images"
        model_path = "big-lama"
        checkpoint_path = "best.ckpt"

        command = [
            "python", "bin/predict.py",
            "--config-path", "../configs/prediction",
            "--config-name", "big-lama.yaml",
            f"+model.path={model_path}",
            f"+model.checkpoint={checkpoint_path}",
            f"+indir={input_dir}",
            f"+outdir={output_dir}",
            "+dataset.img_suffix=.jpg",
            "+dataset.mask_generator_kind=default",
            "+out_key=inpainted"
        ]

        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode == 0:
            print("Inpainting completed successfully.")
            return True
        else:
            print("Error during inpainting:", result.stderr)
            return False
    except Exception as e:
        print("An error occurred:", str(e))

if __name__ == '__main__':
    inpaint_image()
