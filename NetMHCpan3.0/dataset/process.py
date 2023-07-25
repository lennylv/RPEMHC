import pandas as pd
lines = open('MHC_pseudo.dat').readlines()
alleles = [line.strip().split()[0] for line in lines]
pse = [line.strip().split()[1] for line in lines]
mhc_name_dict = {}
for i,j in enumerate(alleles):
    mhc_name_dict[j] = pse[i]
fw = open('NetMHCpan_cv_id.txt','w')
pep = []
allele = []
logic = []
mhc = []
for i in range(5):
    lines = open('c00'+str(i)+'_ba').readlines()
    p = [line.strip().split()[0] for line in lines]
    a = [line.strip().split()[2] for line in lines]
    l = [line.strip().split()[1] for line in lines]
    m = [mhc_name_dict[i] for i in a]
    pep = pep+list(p)
    allele = allele+list(a)
    logic = logic +list(l)
    mhc = mhc+list(m)
    print(len(m))
    cv = ''.join([str(i)+'\n']*len(m))
    fw.write(cv)
print(len(mhc))
out = pd.DataFrame({'pep':pep,'allele':allele,'logic':logic,'mhc':mhc})
out.to_csv('NetMHCpan_data.csv',index=False)

for i in range(5):
    lines1 = open('c00'+str(i)+'_ba').readlines()
    lines2 = open('f00'+str(i)+'_ba').readlines()
    print(len(lines1)+len(lines2))