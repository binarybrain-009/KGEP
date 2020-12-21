# KGEP
A Knowledge Graph based Approach for Mobile Application Recommendation
### Files in the folder

+ datas/
    + entity2id.txt:the mapping from entity names to entity IDs in the KG;the first row represents the number of entities;
    + app_id.txt:all item(app) IDs in the KG;
    + kg.txt:knowledge graph, each row is a triple;
    + TransD.json:results of [transD](https://github.com/thunlp/OpenKE) ;
    + user_app.txt:user, app interaction information(IDs), including positive and negative samples;
    + user_app_kg.txt:user, app interaction information(name);
    + user_id.txt:sets of user IDs;
    + user_similar_user_kg.txt:knowledge graph of similar users.
+ src/:implementations of KGEP.
### Required packages
* python == 3.7.6
* tensorflow == 1.15.2
* numpy == 1.18.3
* sklearn == 0.20.1
### Running the code
```
$ cd src
$ python main.py
```
