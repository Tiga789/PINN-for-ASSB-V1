DATAFOLDER=../Data/my_comsol_npz_assb_discharge

python main.py -nosimp -df $DATAFOLDER -i input_assb_discharge_pretrain
python main.py -nosimp -df $DATAFOLDER -i input_assb_discharge_finetune
python main.py -nosimp -df $DATAFOLDER -i input_assb_discharge_refine
