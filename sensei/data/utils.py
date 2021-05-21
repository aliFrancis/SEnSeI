import numpy as np

DESCRIPTORS = {'Sentinel2':
                    np.array([
                       [ 425. ,  443. ,  461. ],
                       [ 446. ,  494. ,  542. ],
                       [ 537.5,  560. ,  582.5],
                       [ 645.5,  665. ,  684.5],
                       [ 694. ,  704. ,  714. ],
                       [ 731. ,  740. ,  749. ],
                       [ 767. ,  781. ,  795. ],
                       [ 763.5,  834. ,  904.5],
                       [ 847.5,  864. ,  880.5],
                       [ 930.5,  944. ,  957.5],
                       [1337. , 1375. , 1413. ],
                       [1541. , 1612. , 1683. ],
                       [2074. , 2194. , 2314. ]
                       ]),
                'SLSTR':
                    np.array([
                       [ 545  ,  555  ,  565  ],
                       [ 649  ,  659  ,  669  ],
                       [ 855  ,  865  ,  875  ],
                       [1363.5, 1374  , 1384.5],
                       [1570  , 1612  , 1660  ],
                       [2200  , 2250  , 2300  ]
                       ]),
                'Landsat8':
                    np.array([
                       [  435. ,   443. ,   451. ],
                       [  452. ,   482. ,   512. ],
                       [  533.5,   562. ,   590.5],
                       [  636.5,   655. ,   673.5],
                       [  851. ,   865. ,   879. ],
                       [ 1566.5,  1609. ,  1651.5],
                       [ 2114.5,  2201. ,  2287.5],
                       [  496.5,   590. ,   683.5],
                       [ 1363.5,  1374. ,  1384.5],
                       [10600. , 10895. , 11190. ],
                       [11500. , 12005. , 12510. ]
                       ]),
                'SPARCS':
                    np.array([
                       [  435. ,   443. ,   451. ],
                       [  452. ,   482. ,   512. ],
                       [  533.5,   562. ,   590.5],
                       [  636.5,   655. ,   673.5],
                       [  851. ,   865. ,   879. ],
                       [ 1566.5,  1609. ,  1651.5],
                       [ 2114.5,  2201. ,  2287.5], # NO PANCHROMATIC BAND IN THIS DATASET
                       [ 1363.5,  1374. ,  1384.5],
                       [10600. , 10895. , 11190. ],
                       [11500. , 12005. , 12510. ]
                       ]),
                'Landsat7':
                    np.array([
                       [  454.5,   491. ,   527.5],
                       [  519. ,   560. ,   601. ],
                       [  631.5,   662. ,   692.5],
                       [  772. ,   835. ,   898. ],
                       [ 1547. ,  1648. ,  1749. ],
                       [10310. , 11335. , 12360. ],
                       [10310. , 11335. , 12360. ],
                       [ 2064.5,  2205. ,  2345.5],
                       [  514.5,   705. ,   895.5]
                       ]),
                'PeruSat1':
                    np.array([
                       [630,665,700],
                       [530,560,590],
                       [450,475,500],
                       [752,819,885]
                       ])
                }

# SYNTHETIC_DICT holds possible combinations of neighbouring bands in Sentinel-2
#     and Landsat 8, which can be added together to produce "new" bands in order
#     to increase overall variation, and to e.g. help learn Landsat 7's thermal
#     band, which is roughly equivalent to both Landsat 8 thermal bands combined.

SYNTHETIC_DICT = {'Sentinel2':[
         {'name':'Landsat8.B8_SYNTHETIC',
          'type': 'TOA Reflectance',
          'descriptor':[446,560,684.5],
          'inputs':['B02','B03','B04']
          },
         {'name':'Landsat7.B8_SYNTHETIC',
          'type': 'TOA Reflectance',
          'descriptor':[537.5,722,904.5],
          'inputs':['B03','B04','B05','B06','B07','B08']
          },
         {'name':'RedEdgeSum_SYNTHETIC',
          'type': 'TOA Reflectance',
          'descriptor':[694,740,796],
          'inputs':['B05','B06','B07']
          },
         {'name':'BlueGreen_SYNTHETIC',
          'type': 'TOA Reflectance',
          'descriptor':[446,540,582.5],
          'inputs':['B02','B03']
          },
         {'name':'GreenRed_SYNTHETIC',
          'type': 'TOA Reflectance',
          'descriptor':[537.5,600,684.5],
          'inputs':['B03','B04']
          },
         {'name':'AerosolBlue_SYNTHETIC',
          'type': 'TOA Reflectance',
          'descriptor':[425,455,542],
          'inputs':['B01','B02']
          }
          ],
    'Landsat8':[
         {'name':'Landsat7.B6_SYNTHETIC',
          'type': 'TOA Reflectance',
          'descriptor':[10600,11300,12510],
          'inputs':['B10','B11']
          },
         {'name':'BlueGreen_SYNTHETIC',
          'type': 'TOA Reflectance',
          'descriptor':[452,520,590.5],
          'inputs':['B2','B3']
          },
         {'name':'GreenRed_SYNTHETIC',
          'type': 'TOA Reflectance',
          'descriptor':[533.5,610,673.5],
          'inputs':['B3','B4']
          },
         {'name':'AerosolBlue_SYNTHETIC',
          'type': 'TOA Reflectance',
          'descriptor':[435,451.5,512],
          'inputs':['B1','B2']
          }
          ]
    }
