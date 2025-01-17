# Text Guided Generation of Head Avatars

## Training:
The model can be trained by calling:
```
python -m callable.train --save_dir "./save/my_model" --overwrite
```

## Sampling:
The model can be either sampled multiple times by defining a ``.txt`` file with the prompts.
```
python -m callable.sample --model_path "./save/demb_256_nemb_128/model000002000.pt" --input_text "./assets/example_samples.txt"
```
Or by prompting the model directly:
```
python -m callable.sample --model_path "./save/demb_256_nemb_128/model000002000.pt" --text_prompt "A person is yawning"
```

## Evaluate:
The model can be evaluated as follows:
```
python -m callable.evaluate --model_path "./save/demb_256_nemb_128/model000002000.pt" --input_text "./assets/example_samples.txt"
```
