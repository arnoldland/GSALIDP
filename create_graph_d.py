# Function: create contact graph from gro file
import numpy as np
import pandas as pd
import csv

thresh = 1.0 #unit nm
num_files = 35

def main():
    for flag in ['A','B']:
        for k in range(1,num_files+1): 
            for m in range(1,17): 
                t = round(0.0-0.2*(m-1),1)
                path = 'raw_data/monomer/structure/'+str(k)+'_'+str(t)+'ns-'+flag+'.gro'
                df = pd.read_csv(path,
                sep = '[\s]{1,}',
                engine = 'python',
                header = None,
                index_col = False,
                skiprows=2,#skip str 3
                skipfooter=1,#skip end 1
                names= ["residue","atom","atomid","x","y","z"])
                df_CA = df[df['atom'] == 'CA']
                row = []
                col = []
                for i in range(df_CA.iloc[:,0].size):
                    for j in range(df_CA.iloc[:,0].size):
                        p_i = df_CA.iloc[i,3:6]#x y z col
                        p_j = df_CA.iloc[j,3:6]#x y z col
                        distance = np.linalg.norm(p_i - p_j)
                        if distance < thresh :
                            row.append(i)
                            col.append(j)
                filepath = './contact_graph/contactg_'+str(k)+'_'+str(t)+'ns-'+flag+'.csv'
                rows = zip(row,col)
                with open(filepath,"w",newline='') as f:
                    writer = csv.writer(f)
                    for i in rows:
                        writer.writerow(i)
    print("done!")

if __name__ == '__main__':
    main()