import pandas as pd
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA


#FOR SETUP
def iLearnSetupTSV(filepath):
    x = pd.read_csv(filepath, sep='\t')
    # convert times to string first
    x['timeStamp'] = x['timeStamp'].apply(lambda x: str(x))

    # add 00 because datetime is picky
    x['timeStamp'] = x['timeStamp'].apply(lambda x: x + '00')

    # create datetime objects on timestamp through apply and lambda functions
    x['timeStamp'] = x['timeStamp'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f%z'))

    # remove all the unwanted parts of metadata
    x['metaData'] = x['metaData'].str.replace('{', '')
    x['metaData'] = x['metaData'].str.replace('}', '')
    x['metaData'] = x['metaData'].str.replace('"', '')
    x['metaData'] = x['metaData'].str.replace("'", "")

    # split on comma and add to dataframe
   # new = x['metaData'].str.split(',', n=1, expand=True)

    # add split to df
    #x['metaData1'] = new[0]
    #x['metaData2'] = new[1]

    x['metaData1'] = x['metaData'].str.split(',').str.get(0)
    x['metaData2'] = x['metaData'].str.split(',', n = 1).str.get(1)

    x.drop(columns=['metaData'], inplace=True)

    return x

def iLearnSetup(eventdf):
    # remove all the unwanted parts of metadata
    x = eventdf
    x['metaData'] = x['metaData'].str.replace('{', '')
    x['metaData'] = x['metaData'].str.replace('}', '')
    x['metaData'] = x['metaData'].str.replace('"', '')
    x['metaData'] = x['metaData'].str.replace("'", "")

    # split on comma and add to df
    #new = x['metaData'].str.split(',', n=1, expand=True)

    # add split to df
    #x['metaData1'] = new[0]
    #x['metaData2'] = new[1]

    x['metaData1'] = x['metaData'].str.split(',').str.get(0)
    x['metaData2'] = x['metaData'].str.split(',', n = 1).str.get(1)

    x.drop(columns=['metaData'], inplace=True)

    return x


def setupLAexe(LAexedf):
    difrep = {'Easy': 2, 'Very easy': 1, 'Neutral': 3, 'Hard': 4, 'Very hard': 5}
    funrep = {'Fun': 4, 'Very fun': 5, 'Neutral': 3, 'Not fun': 2, 'Very not fun': 1}
    LAexe = LAexedf
    LAexe['difficultyRating'] = LAexe['difficultyRating'].replace(difrep).fillna(0)
    LAexe['funRating'] = LAexe['funRating'].replace(funrep).fillna(0)
    return LAexe

#FOR SCHOOL LEVEL STATS
def teachCount(userdf, sid):
    #count on school id and teacher type
    count = np.sum((userdf.school_id == sid) & (userdf.UserType == 'Teacher'))
    return count

def studentCount(userdf, sid):
    #count on school id and student type
    count = np.sum((userdf.school_id == sid) & (userdf.UserType == 'Student'))
    return count

def classCount(userdf, sid):
    #uses unique to build a list of school classes from a dataframe for a school
    z = pd.DataFrame()
    z = z.append(userdf[(userdf.school_id == sid) & (userdf.UserType == 'Student')])
    count = len(z['schoolClass_id'].unique())
    return count

def learningTrackCount(eventdf, userdf, sid, classid=9999):
    # building dataframe for specific school and class
    if classid != 9999:
        s = pd.DataFrame()
        s = s.append(userdf[(userdf.school_id == sid) & (userdf.schoolClass_id == classid)])

    else:
        s = pd.DataFrame()
        s = s.append(userdf[(userdf.school_id == sid)])

    # building dataframe for specific class actions for learning track and listing all unique options
    x = pd.DataFrame()
    x = x.append(eventdf[(eventdf['originatingUserId'].isin(s.id)) & (eventdf.eventType == 'LearningTrackStarted')])
    x['metaData2'] = x['metaData2'].str.replace('learningTrackId:', '')
    return len(x['metaData2'].unique())


def useLTList(eventdf, userdf, sid, classid=9999):
    # building dataframe for specific school and class
    if classid != 9999:
        s = pd.DataFrame()
        s = s.append(userdf[(userdf.school_id == sid) & (userdf.schoolClass_id == classid)])

    else:
        s = pd.DataFrame()
        s = s.append(userdf[(userdf.school_id == sid)])

    # building dataframe for specific class actions for learning track and listing all unique options
    x = pd.DataFrame()
    x = x.append(eventdf[(eventdf['originatingUserId'].isin(s.id)) & (eventdf.eventType == 'LearningTrackStarted')])
    x['metaData2'] = x['metaData2'].str.replace('learningTrackId:', '')
    return x


def progLT(eventdf, userdf, LAdf, LTdf, linkdf, sid, classid=9999):
    # building dataframe for specific school and class
    if classid != 9999:
        s = pd.DataFrame()
        s = s.append(userdf[(userdf.school_id == sid) & (userdf.schoolClass_id == classid)])

    else:
        s = pd.DataFrame()
        s = s.append(userdf[(userdf.school_id == sid)])

    # Dataframe for learning activities by school
    x = pd.DataFrame()
    x = x.append(eventdf[(eventdf['originatingUserId'].isin(s.id)) & (eventdf.eventType == 'LearningActivityFinished')])
    x = x[x.metaData2 != 'startedFromFreeLearn:true']
    x['metaData2'] = x['metaData2'].str.replace('learningActivityId:', '')
    if x.size == 0:
        print('There are no learning activities completed')
    print(x)

    y = pd.DataFrame()
    y = y.append(eventdf[(eventdf['originatingUserId'].isin(s.id)) & (eventdf.eventType == 'LearningTrackStarted')])
    y['metaData2'] = y['metaData2'].str.replace('learningTrackId:', '')
    # y = y.drop(columns = ['ownerId', 'name', 'description', 'learningTrackCluster_id'])

    LAkey = LAdf.drop(
        columns=['name', 'iconName', 'duration', 'isFreeLearn', 'averageDifficultyRating', 'averageFunRating'])
    LTkey = LTdf
    LTkey = LTkey[LTkey['id'].isin(y.metaData2)]

    plt.close('all')

    for name in LTkey._id:
        try:
            plink = linkdf[linkdf.learningTrack_id == name]
            d = x.merge(LAkey, left_on = 'metaData2', right_on = 'id')
            q = d.merge(plink[['sequence', 'learningActivity_id']], left_on='_id_y', right_on='learningActivity_id')
            print(LTkey[LTkey['_id'] == name]['name'])
            q[['originatingUserId', 'sequence']].groupby(['originatingUserId']).max()['sequence'].plot.hist()
            plt.show()
        except:
            print('No learning track activities completed')

def schoolList(df, sid):
    #uses unique to build a list of school classes from a dataframe for a school
    z = pd.DataFrame()
    z = z.append(df[df.school_id == sid])
    z = z.sort_values(by = ['schoolClass_id', 'UserType'])
    return z

def schoolUseCount(eventdf, userdf):
    l = eventdf.merge(userdf[['school_id', 'id']], left_on = 'originatingUserId', right_on = 'id')
    print(l.groupby('school_id').count())

def sorcEvent(eventdf, userdf, sid, classid = 9999):
    if classid != 9999:
        s = pd.DataFrame()
        s = s.append(userdf[(userdf.school_id == sid) & (userdf.schoolClass_id == classid)])

    else:
        s = pd.DataFrame()
        s = s.append(userdf[(userdf.school_id == sid)])

    # Dataframe for learning activities by school
    x = pd.DataFrame()
    x = x.append(eventdf[(eventdf['originatingUserId'].isin(s.id))])
    return x

#FOR USER LEVEL STATS

def UserCount(eventdf, user, count):
    useru = eventdf[user].unique()
    countu = eventdf[count].unique()
    dictx = {}

    for x in useru:
        for y in countu:
            dictx.update({x + ' ' + y: [x, y, np.sum((eventdf[user] == x) & (eventdf[count] == y))]})

    count = pd.DataFrame.from_dict(dictx, orient='index')

    return count

def laStartCount(finladf, userid):
    v = finladf[(finladf.studentId == userid)]
    return len(v.index)

def laFinCount(finladf, userid):
    v = finladf[finladf.completedTime.notnull()]
    v = v[(v.studentId == userid)]
    return len(v.index)

def laTotTime(finladf, userid):
    v = finladf[finladf.completedTime.notnull()]
    v = v[(v.studentId == userid)]
    v['Time to complete LA'] = v.completedTime - v.created
    t = v['Time to complete LA'].sum()
    return t

def laAvgTime(finladf, userid):
    v = finladf[finladf.completedTime.notnull()]
    v = v[(v.studentId == userid)]
    v['Time to complete LA'] = v.completedTime - v.created

    z = finladf[(finladf.studentId == userid)]

    t = v['Time to complete LA'].sum()
    avg = v['Time to complete LA'].mean()
    # avg = time/len(z.index) choosing to keep mean of completed, while reporting difference between starts and completes

    return avg

def probCount(eventdf, userid):
    v = eventdf[(eventdf.eventType == 'ProblemOnActivitySignaled') & (eventdf.originatingUserId == userid)]
    return len(v.index)

def avgDifRating(laexedf, userid):
    v = laexedf[laexedf['studentId'] == userid]
    avg = v['difficultyRating'].mean()
    return avg

def avgFunRating(laexedf, userid):
    v = laexedf[laexedf['studentId'] == userid]
    avg = v['funRating'].mean()
    return avg

#FOR TEACHER LEVEL STATS

def lt_add_count(eventdf, userid):
    v = eventdf[(eventdf.eventType == 'LearningTrackAddedToOwnLibrary') & (eventdf.originatingUserId == userid)]
    return len(v.index)

def lt_edit_count(eventdf, userid):
    v = eventdf[(eventdf.eventType == 'LearningTrackEdited') & (eventdf.originatingUserId == userid)]
    return len(v.index)

def group_creation_count(eventdf, userid):
    v = eventdf[(eventdf.eventType == 'GroupCreated') & (eventdf.originatingUserId == userid)]
    return len(v.index)

def group_manipulation_count(eventdf, userid):
    v = eventdf[(eventdf.eventType.isin(['MembersAddedToGroup', 'MembersRemovedFromGroup'])) &
                (eventdf.originatingUserId == userid)]
    return len(v.index)

def total_lt_assign(groupmemdf, ltexedf, userid):
    x = groupmemdf[groupmemdf['userId'] == userid]
    y = groupmemdf[groupmemdf['group_id'].isin(x['group_id'])]
    v = ltexedf[ltexedf['studentId'].isin(y['userId'])]
    return len(v['learningTrack_id'].unique())

def avg_lt_assign(groupmemdf, ltexedf, userid):
    g = groupmemdf[(groupmemdf['GroupRole'] == 'Owner') & (groupmemdf['userId'] == userid)]
    num_group = len(g.index)
    num_lt = total_lt_assign(groupmemdf, ltexedf, userid)
    if num_group == 0:
        return np.nan
    else:
        avg = num_lt/num_group
        return avg

#FOR MACHINE LEARNING

def elbowsseDiagram(dataset, krange):
    sse = []
    for k in range(1, krange):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(dataset)
        sse.append(kmeans.inertia_)

    plt.plot(range(1, krange), sse)
    plt.xticks(range(1, krange))
    plt.title('Elbow Diagram')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.show()


def silDiagram(dataset, krange):
    sil = []
    for k in range(2, krange):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(dataset)
        score = silhouette_score(dataset, kmeans.labels_)
        sil.append(score)

    plt.plot(range(2, krange), sil)
    plt.xticks(range(1, krange))
    plt.title('Silhouette Diagram')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Coefficient")
    plt.show()

def bestEps(dataset, n_neigh):
    neigh = NearestNeighbors(n_neighbors= n_neigh)
    nbrs = neigh.fit(dataset)
    distances, indices = nbrs.kneighbors(dataset)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]

    plt.figure(figsize=(20,10))
    plt.plot(distances)
    plt.show()

