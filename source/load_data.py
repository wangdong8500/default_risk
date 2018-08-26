import pandas as pd
from preprocessing import application_train_test, bureau_and_balance, previous_applications, pos_cash, installments_payments, credit_card_balance
from glob import glob
import time
from contextlib import contextmanager
import gc

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

def load_data(dataDir):
    if 'full_data.csv' in [f[2:] for f in glob("./*.csv")]:
        print('Data already existed.')
        return pd.read_csv('full_data.csv')
    else:
        df = application_train_test(dataDir)
        with timer("Process bureau and bureau_balance"):
            bureau = bureau_and_balance(dataDir)
            print("Bureau df shape:", bureau.shape)
            df = df.join(bureau, how='left', on='SK_ID_CURR')
            del bureau
            gc.collect()
        with timer("Process previous_applications"):
            prev = previous_applications(dataDir)
            print("Previous applications df shape:", prev.shape)
            df = df.join(prev, how='left', on='SK_ID_CURR')
            del prev
            gc.collect()
        with timer("Process POS-CASH balance"):
            pos = pos_cash(dataDir)
            print("Pos-cash balance df shape:", pos.shape)
            df = df.join(pos, how='left', on='SK_ID_CURR')
            del pos
            gc.collect()
        with timer("Process installments payments"):
            ins = installments_payments(dataDir)
            print("Installments payments df shape:", ins.shape)
            df = df.join(ins, how='left', on='SK_ID_CURR')
            del ins
            gc.collect()
        with timer("Process credit card balance"):
            cc = credit_card_balance(dataDir)
            print("Credit card balance df shape:", cc.shape)
            df = df.join(cc, how='left', on='SK_ID_CURR')
            del cc
            gc.collect()
        df.to_csv('full_data.csv', index= False)
        return df