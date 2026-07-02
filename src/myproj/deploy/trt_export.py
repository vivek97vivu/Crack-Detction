import torch
import os
import argparse

def export_to_onnx(model_class, checkpoint_path, output_onnx_path, input_shape):
    """
    Exports a PyTorch model checkpoint to ONNX format.
    """
    print(f"Loading model checkpoint from {checkpoint_path}...")
    device = torch.device("cpu")
    
    # Instantiate model
    model = model_class()
    
    # Load weights
    state_dict = torch.load(checkpoint_path, map_location=device)
    if 'model' in state_dict:
        state_dict = state_dict['model']
    elif 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    
    # Generate dummy input
    dummy_input = torch.randn(*input_shape, device=device)
    
    print(f"Exporting to ONNX at {output_onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Export complete!")

def compile_to_tensorrt(onnx_path, trt_path, fp16=True):
    """
    Executes trtexec command line to compile the ONNX model to TensorRT.
    """
    print(f"Compiling {onnx_path} to TensorRT engine at {trt_path}...")
    
    # Build command string
    cmd = f"trtexec --onnx={onnx_path} --saveEngine={trt_path}"
    if fp16:
        cmd += " --fp16"
        
    print(f"Execution command: {cmd}")
    # We print instructions because trtexec requires Jetson environment (JetPack)
    print("\n[Jetson Setup Instruction]")
    print("Ensure you have JetPack installed on your NVIDIA Jetson.")
    print("Run the command in your shell:")
    print(f"  /usr/src/tensorrt/bin/{cmd}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export models to ONNX and TensorRT")
    parser.add_argument("--model-type", type=str, required=True, choices=["gate", "segmenter"], 
                        help="Which model to export (gate or segmenter)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to PyTorch checkpoint (.pth)")
    parser.add_argument("--output-dir", type=str, default="runs/deploy", help="Output directory")
    parser.add_argument("--fp16", action="store_true", help="Compile TensorRT engine using FP16 precision")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.model_type == "gate":
        from myproj.inference.gate import GateClassifier
        model_class = GateClassifier
        input_shape = (1, 3, 224, 224)
        name = "gate_classifier"
    else:
        from myproj.inference.segmenter import UNet
        model_class = UNet
        input_shape = (1, 3, 256, 256)
        name = "segmenter_unet"
        
    onnx_path = os.path.join(args.output_dir, f"{name}.onnx")
    trt_path = os.path.join(args.output_dir, f"{name}.engine")
    
    export_to_onnx(model_class, args.checkpoint, onnx_path, input_shape)
    compile_to_tensorrt(onnx_path, trt_path, fp16=args.fp16)
