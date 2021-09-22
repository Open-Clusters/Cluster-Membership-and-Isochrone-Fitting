##############################     PROPER MOTION PROGRAM STARS     ##############################

# import pandas as pd
# from sklearn.mixture import GaussianMixture
# from matplotlib import pyplot as plt

# #importing the data file
# sample = pd.read_csv("NGC2112.csv")
# print("Shape of Dataset: ", sample.shape)

# #Drop rows that have no values
# ####################
# sample.dropna(inplace = True)
# print('After dropping rows that contain Nan: ', sample.shape)
# sample.to_csv('sample.csv')
# ####################
# #selecting desired columns only
# ####################
# data = sample[['ra', 'dec', 'parallax','pmra', 'pmdec']]
# #print(data.head(2))
# ####################
# #Normalizing the data
# def normalize(dataset):
#     dataNorm=(dataset-dataset.median())/dataset.std()
#     #dataNorm["id"]=dataset["id"]
#     return dataNorm

# data=normalize(data)
# print(data.sample(5))
# ####################
# #fitting the Gaussian model to the sample data
# gmm = GaussianMixture(n_components=2, covariance_type='full', init_params="kmeans")
# gmm.fit(data)
# #print(gmm.means_)
# #print(gmm.covariances_)
# #labels = gmm.predict(data)
# mem_prob = gmm.predict_proba(data)
# #prob = mem_prob.tolist()
# prob= [comp[1] for comp in mem_prob]
# frame =pd.DataFrame(sample)
# #frame['cluster'] = labels
# frame['prob']= prob
# labels = []
# for p in prob:    
#     if p >= 0.8:
#         labels.append(1)
#     else:
#         labels.append(0)
# frame['cluster'] = labels
# frame.to_csv('2243_3.5output.csv')
# ###################
# #sorting cluster members and field stars to two different files from the output file
# df = pd.read_csv("2243_3.5output.csv")
# df= df[["source_id", "ra", "dec", "parallax", "pmra", "pmdec", "bp_rp", "phot_g_mean_mag", "cluster", "prob"]]
# print(df['prob'])
# mem_stars = df.loc[df['prob'] >= 0.8]
# print(len(mem_stars))
# mem_stars.to_csv("2243_3.5cluster.csv")
# field_stars = df.loc[df['prob'] < 0.8]
# field_stars.to_csv("2243_3.5field.csv")
# count = len(mem_stars)
# ###################
# plt.figure(figsize=(6,5))
# color=['blue', 'red']
# for i in range(0,2):
#    data = frame[frame["cluster"]==i]  
#     #plt.scatter(data["ra"],data["dec"], s =0.2, c=color[i])
#  #  plt.scatter(data["pmra"],data["pmdec"], s=0.4, c=color[i])
#     #plt.scatter(data["bp_rp"],data["phot_g_mean_mag"], s = 0.9, c=color[i], alpha=0.6)
#    plt.scatter(data['phot_g_mean_mag'], data["prob"], s= 2**2, c=color[i])
# #plt.xlim(-18, 18)
# #plt.ylim(-18, 18)
# plt.title("VPD of NGC 2243")
# plt.xlabel("PMRA (mas/yr)")
# plt.ylabel("PMDEC (mas/yr)")
# #plt.suptitle("CMD of NGC6791")
# #plt.xlabel("bp_rp")
# #plt.ylabel("g_mag")
# #plt.grid()
# #plt.gca().invert_yaxis()
# #plt.legend()
# #plt.savefig('CMD_NGC188.png')
# plt.show();

##############################     PROPER MOTION PROGRAM ENDS     ##############################


##############################     ISOCHRONE FITTING PROGRAM STARTS     ##############################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

isochrone_files = ["2243_iso_3.2_0.0080.dat",
                   "2243_iso_3.4_0.0080.dat", 
                   "2243_iso_3.6_0.0080.dat", 
                   "2243_iso_3.2_0.0090.dat", 
                   "2243_iso_3.4_0.0090.dat",
                   "2243_iso_3.6_0.0090.dat"]
#################################################

cluster_file = "2243_3.5cluster.csv"
#####################################

isochrone1 = np.loadtxt(isochrone_files[0], usecols=(-3,-2,-1)) 
isochrone2 = np.loadtxt(isochrone_files[1], usecols=(-3,-2,-1))
isochrone3 = np.loadtxt(isochrone_files[2], usecols=(-3,-2,-1))
isochrone4 = np.loadtxt(isochrone_files[3], usecols=(-3,-2,-1))
isochrone5 = np.loadtxt(isochrone_files[4], usecols=(-3,-2,-1))
isochrone6 = np.loadtxt(isochrone_files[5], usecols=(-3,-2,-1))

iso_list1 = isochrone1.tolist()
df_isochrone1 = pd.DataFrame(iso_list1, columns=["Gmag", "G_BP", "G_RP"])

iso_list2 = isochrone2.tolist()
df_isochrone2 = pd.DataFrame(iso_list2, columns=["Gmag", "G_BP", "G_RP"])

iso_list3 = isochrone3.tolist()
df_isochrone3 = pd.DataFrame(iso_list3, columns=["Gmag", "G_BP", "G_RP"])

iso_list4 = isochrone4.tolist()
df_isochrone4 = pd.DataFrame(iso_list4, columns=["Gmag", "G_BP", "G_RP"])

iso_list5 = isochrone5.tolist()
df_isochrone5 = pd.DataFrame(iso_list5, columns=["Gmag", "G_BP", "G_RP"])

iso_list6 = isochrone6.tolist()
df_isochrone6 = pd.DataFrame(iso_list6, columns=["Gmag", "G_BP", "G_RP"])

dfs = [df_isochrone1, df_isochrone2, df_isochrone3, df_isochrone4, df_isochrone5, df_isochrone6]

###___UPPER PANEL ISOCHRONE DATA___###
reddening1 = 0.070
dist_mod1 = 13.1

###___LOWER PANEL ISOCHRONE DATA___###
reddening2 = 0.050
dist_mod2 = 13.0

df = pd.read_csv(cluster_file)
df_cluster = df.filter(items=["phot_g_mean_mag", "bp_rp"])

fig, ax = plt.subplots(figsize=(13, 9))

scatter = plt.scatter(df_cluster["bp_rp"], df_cluster["phot_g_mean_mag"], s=0.3, c="red", label="CMD")

ax.plot((df_isochrone1["G_BP"]-df_isochrone1["G_RP"])+reddening1, df_isochrone1["Gmag"]+dist_mod1, label="Age: 3.2Gyr, Z: 0.0080,\nE(B-V)=%.4f, M=%.2f" %(reddening1, dist_mod1))
ax.plot((df_isochrone2["G_BP"]-df_isochrone2["G_RP"])+reddening1, df_isochrone2["Gmag"]+dist_mod1, label="Age: 3.4Gyr, Z: 0.0080,\nE(B-V)=%.4f, M=%.2f" %(reddening1, dist_mod1))
ax.plot((df_isochrone3["G_BP"]-df_isochrone3["G_RP"])+reddening1, df_isochrone3["Gmag"]+dist_mod1, label="Age: 3.6Gyr, Z: 0.0080,\nE(B-V)=%.4f, M=%.2f" %(reddening1, dist_mod1))
ax.plot((df_isochrone4["G_BP"]-df_isochrone4["G_RP"])+reddening2, df_isochrone4["Gmag"]+dist_mod2, label="Age: 3.2Gyr, Z: 0.0090,\nE(B-V)=%.4f, M=%.2f" %(reddening2, dist_mod2))
ax.plot((df_isochrone5["G_BP"]-df_isochrone5["G_RP"])+reddening2, df_isochrone5["Gmag"]+dist_mod2, label="Age: 3.4Gyr, Z: 0.0090,\nE(B-V)=%.4f, M=%.2f" %(reddening2, dist_mod2))
ax.plot((df_isochrone6["G_BP"]-df_isochrone6["G_RP"])+reddening2, df_isochrone6["Gmag"]+dist_mod2, label="Age: 3.6Gyr, Z: 0.0090,\nE(B-V)=%.4f, M=%.2f" %(reddening2, dist_mod2))
########################################################################################################

plt.suptitle("NGC2243 isochrones")
plt.gca().invert_yaxis()
plt.legend(loc="lower left", fontsize = 'xx-small')
plt.xlabel("BP-RP")
plt.ylabel("M")

fig1, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3, sharex=True, sharey=True, figsize=(13,9))
fig1.text(0.5, 0.04, 'BP-RP', ha='center', va='center')
fig1.text(0.06, 0.5, 'M', ha='center', va='center', rotation='vertical')

# ax.plot wali line me label ke parameter ki values change krni hain (e.g A: 3.2B, M: 0.0152) A for age M for metallicity 
ax1.plot((df_isochrone1["G_BP"]-df_isochrone1["G_RP"])+reddening1, df_isochrone1["Gmag"]+dist_mod1, label="Age: 3.2Gyr, Z: 0.0080,\nE(B-V)=%.4f, M=%.2f" %(reddening1, dist_mod1))
ax1.scatter(df_cluster["bp_rp"], df_cluster["phot_g_mean_mag"], s=0.3, c="red", label="CMD")

ax2.plot((df_isochrone2["G_BP"]-df_isochrone2["G_RP"])+reddening1, df_isochrone2["Gmag"]+dist_mod1, label="Age: 3.4Gyr, Z: 0.0080,\nE(B-V)=%.4f, M=%.2f" %(reddening1, dist_mod1))
ax2.scatter(df_cluster["bp_rp"], df_cluster["phot_g_mean_mag"], s=0.3,c="red", label="CMD")

ax3.plot((df_isochrone3["G_BP"]-df_isochrone3["G_RP"])+reddening1, df_isochrone3["Gmag"]+dist_mod1, label="Age: 3.6Gyr, Z: 0.0080,\nE(B-V)=%.4f, M=%.2f" %(reddening1, dist_mod1))
ax3.scatter(df_cluster["bp_rp"], df_cluster["phot_g_mean_mag"], s=0.3,c="red", label="CMD")
            
ax4.plot((df_isochrone4["G_BP"]-df_isochrone4["G_RP"])+reddening2, df_isochrone4["Gmag"]+dist_mod2, label="Age: 3.2Gyr, Z: 0.0090,\nE(B-V)=%.4f, M=%.2f" %(reddening2, dist_mod2))
ax4.scatter(df_cluster["bp_rp"], df_cluster["phot_g_mean_mag"], s=0.3,c="red", label="CMD")

ax5.plot((df_isochrone5["G_BP"]-df_isochrone5["G_RP"])+reddening2, df_isochrone5["Gmag"]+dist_mod2, label="Age: 3.4Gyr, Z: 0.0090,\nE(B-V)=%.4f, M=%.2f" %(reddening2, dist_mod2))
ax5.scatter(df_cluster["bp_rp"], df_cluster["phot_g_mean_mag"], s=0.3,c="red", label="CMD")

ax6.plot((df_isochrone6["G_BP"]-df_isochrone6["G_RP"])+reddening2, df_isochrone6["Gmag"]+dist_mod2, label="Age: 3.6Gyr, Z: 0.0090,\nE(B-V)=%.4f, M=%.2f" %(reddening2, dist_mod2))
ax6.scatter(df_cluster["bp_rp"], df_cluster["phot_g_mean_mag"], s=0.3,c="red", label="CMD")
#############################################################################
ax1.legend(loc="lower left", fontsize = 'xx-small')
ax2.legend(loc="lower left", fontsize = 'xx-small')
ax3.legend(loc="lower left", fontsize = 'xx-small')
ax4.legend(loc="lower left", fontsize = 'xx-small')
ax5.legend(loc="lower left", fontsize = 'xx-small')
ax6.legend(loc="lower left", fontsize = 'xx-small')

plt.suptitle("NGC2243 Isochrones")
##################################################
ax1.invert_yaxis()
plt.show()
##############################     ISOCHRONE FITTING PROGRAM ENDS     ##############################

##############################         CMD PLOTS PROGRAM STARTS      ##################################
sample_file = "2243_3.5sample.csv"
df = pd.read_csv(sample_file)
df_sample = df.filter(items=["phot_g_mean_mag", "bp_rp"])

cluster_file = "2243_3.5cluster.csv"
df = pd.read_csv(cluster_file)
df_cluster = df.filter(items=["phot_g_mean_mag", "bp_rp"])

field_file= "2243_3.5field.csv"
df = pd.read_csv(field_file)
df_field = df.filter(items=["phot_g_mean_mag", "bp_rp"])

fig1, (ax1,ax2,ax3) = plt.subplots(1,3, sharex=True, sharey=True, figsize=(10,5))
fig1.text(0.5, 0.04, 'BP-RP', ha='center', va='center')
fig1.text(0.06, 0.5, 'M', ha='center', va='center', rotation='vertical')

ax1.scatter(df_sample["bp_rp"], df_sample["phot_g_mean_mag"], s=0.3, c="black", label="CMD of sample stars")
ax2.scatter(df_cluster["bp_rp"], df_cluster["phot_g_mean_mag"], s=0.3, c="red", label="CMD of cluster stars")
ax3.scatter(df_field["bp_rp"], df_field["phot_g_mean_mag"], s=0.3, c="blue", label="CMD of field stars")
plt.suptitle("Colour-Magnitude Diagram of NGC2243")
ax1.set_title('Sample Stars')
ax2.set_title('Cluster Stars')
ax3.set_title('Field Stars')
ax1.invert_yaxis()
plt.show()
##############################         CMD PLOTS PROGRAM ENDS      ##################################
