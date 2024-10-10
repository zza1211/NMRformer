from net import *
import numpy as np
import pandas as pd
import math
import os
class dataGene():
    def __init__(self,allMeta):
        # super.__init__()
        self.allmeta=allMeta
    def test_m(self,model_pth,spectra,peaks):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        nmrf = NMRformer(
            input_dim = 128,
            num_classes = 72,
            dim = 256,
            mlp_dim = 512
        ).to(device)
        nmrf.load_state_dict(torch.load(model_pth))
        nmrf=nmrf.eval()
        sorted_peak=sorted(peaks)
        out_csv=pd.DataFrame({'peak':sorted_peak})
        peak_emb=[]
        # n_peaks=[]
        n_pos_peak=[]
        con=np.ones((len(sorted_peak),len(sorted_peak)))
        for j in range(len(sorted_peak)):
            # n_peaks.append(sorted_peak[j])
            sub_spe=np.array(spectra[int(sorted_peak[j]*5000)-64:int(sorted_peak[j]*5000)+64])
            peak=(sorted_peak[j]/12)*100
            peak_emb.append(sub_spe)
            n_pos_peak.append(peak)
            for ci in range(j+1,len(sorted_peak)):
                conc=spectra[int(sorted_peak[j]*5000)]/spectra[int(sorted_peak[ci]*5000)]
                if conc>1:conc=1/conc
                con[j,ci]=conc
                con[ci,j]=conc
        peak_emb=np.array(peak_emb)
        n_pos_peak=np.array(n_pos_peak)
        pe=np.zeros((len(n_pos_peak),128))
        div_term=np.exp(np.arange(0,128,2)*(-math.log(10000)/128))
        pe[:,0::2]=np.sin(n_pos_peak[:, None]*div_term[None, :])
        pe[:,1::2]=np.cos(n_pos_peak[:, None]*div_term[None, :])
        peak_emb=peak_emb+pe
        input=torch.Tensor(peak_emb).unsqueeze(0)
        con=torch.Tensor(con).unsqueeze(0)
        output1=nmrf(input.cuda(),con.cuda())
        prob=torch.softmax(output1[0],dim=1)
        valuesp, indicesp = torch.topk(prob, k=3,dim=1)
        out_csv['pred_0']=self.indices_to_meta(indicesp[0][0].tolist())
        out_csv['prob_0']=valuesp[0][0].tolist()
        out_csv['pred_1']=self.indices_to_meta(indicesp[0][1].tolist())
        out_csv['prob_1']=valuesp[0][1].tolist()
        out_csv['pred_2']=self.indices_to_meta(indicesp[0][2].tolist())
        out_csv['prob_2']=valuesp[0][2].tolist()
        return out_csv



    def indices_to_meta(self,indices):
        meta=[]
        for i in indices:
            meta.append(self.allmeta[i-1])
        return meta
