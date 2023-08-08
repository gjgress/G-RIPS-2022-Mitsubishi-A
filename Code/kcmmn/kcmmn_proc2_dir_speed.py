#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 12:43:56 2023

@author: rileywilde
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 11:25:01 2023

@author: rileywilde
"""




import pandas as pd
import os
import numpy as np
from pyproj import Transformer


def knn_lines(traj,verts,edges,k):
    #this is bloody exhaustive...
    #let's compute once and save forever
    dvec = verts[edges[:,0],:] - verts[edges[:,1],:]
    dvec = dvec/(np.sqrt(np.sum(dvec*dvec,1))[:,None]+1e-6)
    
    x = verts[edges[:,0],0,None]-traj[None,:,0]
    y = verts[edges[:,0],1,None]-traj[None,:,1]
    
    D = np.abs(dvec[:,0,None]*y - dvec[:,1,None]*x)
    
    q = np.argsort(D,axis=0)[:k,:]
    
    ii = np.array(k*[np.arange(q.shape[1]).tolist()])
    
    Dk = D[q,ii]
    
    return q.T,Dk.T


def terrible_approximation_of_velocity(xyt):    
    vel = (xyt[2:,:2]-xyt[0:-2,:2])/(xyt[2:,2]-xyt[0:-2,2])[:,None]
    # vel = (xyt[2:,:2]-xyt[0:-2,:2])/2
    
    v0 = ((xyt[1,:2]-xyt[0,:2])/(xyt[1,2]-xyt[0,2]))[None,:]
    
    vend = ((xyt[-1,:2]-xyt[-2,:2])/(xyt[-1,2]-xyt[-2,2]))[None,:]
    
    vel = np.concatenate((v0,vel,vend),0)
    
    vmag = np.sqrt(np.sum(vel**2,1))
    
    vdir = np.arctan2(vel[:,1],vel[:,0])
    vdir[vdir<0] = vdir[vdir<0] + 2*np.pi
    
    return vmag,vdir
    


root = './data/kcmmn/old'
fnames = [k.zfill(8) for k in  np.arange(0,100).astype(str)] #really just first part

outpath = './data/kcmmn/train'
if os.path.exists(outpath)==False:
    os.mkdir(outpath)



transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857")


make_undirected = False #switch to false for full directed road network
compute_knn_edges = False

k = 5


for f in fnames:
    print('processing: ',f)
    map_edges = pd.read_csv(os.path.join(root,f+'.arcs'), header=None, delimiter='\t').values
    map_nodes = pd.read_csv(os.path.join(root,f+'.nodes'), header=None, delimiter='\t').values
    gps_xyt = pd.read_csv(os.path.join(root,f+'.track'), header=None, delimiter='\t').values
    gt_id = pd.read_csv(os.path.join(root,f+'.route'), header=None, delimiter='\t').values
    
    
    #clean the graph:
    #the graph is currently directed. Let's make it undirected:
       
    b4 = map_edges.shape[0]

    if make_undirected:
        map_edges = np.sort(map_edges,1) 
        undir_str = '_undir'
    else:
        undir_str = '_dir'
        
    map_edges,b = np.unique(map_edges,axis=0,return_inverse=True) #there are some self-loops
    
    i2 = map_edges[:,0]!=map_edges[:,1]
    map_edges = map_edges[i2,:]
    
    ind = np.arange(map_edges.shape[0])
    gt_id = ind[b[gt_id]]
    
    ind2 = map_edges.shape[0]*np.ones(i2.shape[0],int) #did this so it will crash if my reasoning is wrong
    ind2[i2] = np.arange(map_edges.shape[0]) #downshift indices to account for deleted entries
    
    gt_id = ind2[gt_id[i2[gt_id]]] #this is how we do it... ay ay ay
    
    
    

    print(b4, map_edges.shape[0])
    
    x1,x2 = transformer.transform(map_nodes[:,1],map_nodes[:,0])
    
    mn = map_nodes.copy()
    mn[:,0] =x1
    mn[:,1] =x2
    
    x1,x2 = transformer.transform(gps_xyt[:,1],gps_xyt[:,0])
    
    xyt = gps_xyt.copy()
    xyt[:,0] = x1
    xyt[:,1] = x2
    
    
    #RESET NAMES ***BEFORE*** COMPUTING DISTANCES
    map_nodes= mn
    gps_xyt = xyt
    
    
    
    vmag,vdir = terrible_approximation_of_velocity(gps_xyt)
    
    
    
    if (np.isinf(xyt)+np.isnan(xyt)).any() + (np.isinf(mn)+np.isnan(mn)).any():
        print('bad at file ',f)
        
    else:
        if compute_knn_edges:
            knn_edges,knn_dist = knn_lines(gps_xyt[:,:2],map_nodes,map_edges,k)
        else:
            knn_edges,knn_dist = 0,0
        
        
        np.savez( os.path.join(outpath,f+undir_str+'.npz'),map_edges=map_edges,map_nodes=map_nodes,gps_xyt=gps_xyt,gt_id=gt_id,knn_edges=knn_edges, knn_dist=knn_dist,vmag=vmag,vdir=vdir)
    
    
def edge_to_point(Q,D):
    
    l = []
    lens = []
    for i in range(map_edges.shape[0]):
        l.append(np.where(knn_edges==i)[0])
        lens.append(len(l[i]))
        
    lens = np.array(lens)
    
    nonzero_id = np.where(lens>0)[0]
    arr = [l[i] for i in nonzero_id]

    