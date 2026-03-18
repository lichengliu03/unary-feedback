conda create -n webshopmini python=3.9 -y
conda activate webshopmini
conda install -c pytorch faiss-cpu -y
sudo apt update
sudo apt install default-jdk
pip install -r requirements.txt

# setup spacy data (required)
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg

# Download example data (optional, this is already included in the repo)
mkdir -p data
cd data
gdown https://drive.google.com/uc?id=1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib
gdown https://drive.google.com/uc?id=1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu
gdown https://drive.google.com/uc?id=14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O
cd ..

# setup search engine (optional)
python setup_search_engine.py
cd search_engine
bash ../scripts/run_indexing.sh
cd ..
