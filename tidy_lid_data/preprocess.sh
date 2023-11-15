
# Stage 1: Download dataset
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1zslKQwadZaYWXAmfBCvlos9BVQ9k6PHT' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1zslKQwadZaYWXAmfBCvlos9BVQ9k6PHT" -O sixth_edition.zip && rm -rf /tmp/cookies.txt

unzip sixth_edition.zip
rm sixth_edition.zip

# Stage 2: Preprocess
python3 tidy_data.py --data_dir ./sixth_edition --out_dir $1
