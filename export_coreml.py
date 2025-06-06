#!/usr/bin/env python3
"""Export CoreML model - IDE debuggable version of the command line converter.

Replicates this command:
python -m anemll.ane_converter.qwen_converter \
    --model ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455/ \
    --prefix "test_qwen" \
    --batch-size 2 \
    --context-length 256 \
    --output /tmp/qwen-test/float32/
"""

import os
from anemll.ane_converter.qwen_converter import test_conversion

def main():
    """Export CoreML model using the same parameters as the command line."""
    
    print("🚀 Starting CoreML export (IDE debuggable version)")
    print("=" * 60)
    
    # Parameters that match the command line exactly
    model_path = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455/")
    prefix = "test_qwen"
    batch_size = 2
    context_length = 256
    output_dir = "/tmp/qwen-test/float32/"
    lut_bits = None  # Not specified in command line
    
    print("📋 Conversion Parameters:")
    print(f"  Model path: {model_path}")
    print(f"  Prefix: {prefix}")
    print(f"  Batch size: {batch_size}")
    print(f"  Context length: {context_length}")
    print(f"  Output directory: {output_dir}")
    print(f"  LUT bits: {lut_bits}")
    print()
    
    try:
        # Call the same function used by the command line
        result = test_conversion(
            model_path=model_path,
            prefix=prefix,
            context_length=context_length,
            lut_bits=lut_bits,
            batch_size=batch_size,
            output_dir=output_dir,
        )
        
        print(f"\n✅ Conversion completed successfully!")
        print(f"Model type: {type(result)}")
        print(f"Output saved to: {output_dir}{prefix}.mlpackage")
        
    except Exception as e:
        print(f"\n❌ Error during conversion: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main() 