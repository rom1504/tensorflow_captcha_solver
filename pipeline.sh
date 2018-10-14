rm -rf generated_captcha_images2/* extracted_letter_images/*
python3 generate_image.py
python3 extract_single_letters_from_captchas.py
python3 train_model.py
python3 solve_captchas_with_model.py