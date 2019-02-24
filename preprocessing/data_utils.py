import struct
import numpy as np
from PIL import Image, ImageEnhance

# Download the ETL character database files at http://etlcdb.db.aist.go.jp
# put the files ETL8B2C1 etc into ETL8B folder


def read_record(f):
    s = f.read(512)
    r = struct.unpack('>2H4s504s', s)
    i1 = Image.frombytes('1', (64, 63), r[3], 'raw')
    img_out = r + (i1,)
    return img_out


def process_ETL_data(dataset, categories):
    # combine three ETL files together

    name_base = 'ETL8B/ETL8B2C'
    filename = name_base + str(dataset)

    new_img = Image.new('1', (64, 64))

    X, Y = [], []

    for id_category in categories:
        with open(filename, 'rb') as f:
            f.seek((id_category * 160 + 1) * 512)

            for i in range(160):
                try:

                    # start outputting records
                    r = read_record(f)
                    new_img.paste(r[-1], (0, 0))
                    iI = Image.eval(new_img, lambda x: not x)

                    outData = np.asarray(iI.getdata()).reshape(64, 64)

                    X.append(outData)
                    Y.append(r[1])

                except:
                    break
    output = []

    X, Y = np.asarray(X, dtype=np.int32), np.asarray(Y, dtype=np.int32)
    output += [X]
    output += [Y]

    return output
