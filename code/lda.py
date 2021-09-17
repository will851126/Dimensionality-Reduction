import pandas as pd 
import plotly.express as px # for data visualization

from sklearn.preprocessing import StandardScaler # for data standardization
from sklearn.metrics import classification_report # for model evaluation metrics
from sklearn.preprocessing import OrdinalEncoder # to encode categorical variables
from sklearn.tree import DecisionTreeClassifier # for decision tree models

# Sklearn dimensionality reduction
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA # for LDA analysis
from sklearn.decomposition import PCA # for PCA analysis

df=pd.read_csv('/Users/huangbowei/Desktop/coding/Python/ Dimensionality Reduction/data/Real estate.csv')

df.head()

df['Price Band'] = pd.qcut(df['Y house price of unit area'], 3, labels=['1.Affordable (bottom 33%)', '2.Mid-range (middle 33%)', '3.Expensive (top 33%)'])

df['Price Band'].value_counts().sort_index()

enc=OrdinalEncoder()

df['Price Band enc']=enc.fit_transform(df[['Price Band']])

pd.crosstab(df['Price Band'],df['Price Band enc'],margins=False)

# Create a 3D scatter plot
fig = px.scatter_3d(df, 
                    x=df['X1 transaction date'], y=df['X2 house age'], z=df['X3 distance to the nearest MRT station'],
                    color=df['Price Band'],
                    color_discrete_sequence=['#636EFA','#EF553B','#00CC96'], 
                    hover_data=['X3 distance to the nearest MRT station', 'Y house price of unit area', 'Price Band enc'],
                    height=900, width=900
                   )

# Update chart looks
fig.update_layout(#title_text="Scatter 3D Plot",
                  showlegend=True,
                  legend=dict(orientation="h", yanchor="top", y=0, xanchor="center", x=0.5),
                  scene_camera=dict(up=dict(x=0, y=0, z=1), 
                                        center=dict(x=0, y=0, z=-0.2),
                                        eye=dict(x=-1.5, y=1.5, z=0.5)),
                                        margin=dict(l=0, r=0, b=0, t=0),
                  scene = dict(xaxis=dict(backgroundcolor='white',
                                          color='black',
                                          gridcolor='#f0f0f0',
                                          title_font=dict(size=10),
                                          tickfont=dict(size=10),
                                         ),
                               yaxis=dict(backgroundcolor='white',
                                          color='black',
                                          gridcolor='#f0f0f0',
                                          title_font=dict(size=10),
                                          tickfont=dict(size=10),
                                          ),
                               zaxis=dict(backgroundcolor='lightgrey',
                                          color='black', 
                                          gridcolor='#f0f0f0',
                                          title_font=dict(size=10),
                                          tickfont=dict(size=10),
                                         )))


# Update marker size
fig.update_traces(marker=dict(size=2))

# fig.show()

X=df[['X1 transaction date', 'X2 house age', 'X3 distance to the nearest MRT station']]
y=df['Price Band enc'].values


# Get scaler

scaler=StandardScaler()
x_std=scaler.fit_transform(X)



pca=PCA(n_components=2)

X_trans_pca=pca.fit_transform(x_std)
#print('*************** PCA Summary ***************')
#print('No. of features: ', pca.n_features_)
#print('No. of samples: ', pca.n_samples_)
#print('No. of components: ', pca.n_components_)
#print('Explained variance ratio: ', pca.explained_variance_ratio_)



## Performing Linear Discriminant Analysis (LDA)

lda=LDA(
    solver='eigen',
    n_components=2,
)

X_trans_lda=lda.fit_transform(x_std,y)

print('Classes: ', lda.classes_)
print('Priors: ', lda.priors_)
print('Explained variance ratio: ', lda.explained_variance_ratio_)


def fitting(X_in, y, criterion, splitter, mdepth, clweight, minleaf):
    model = DecisionTreeClassifier(criterion=criterion, 
                                   splitter=splitter, 
                                   max_depth=mdepth,
                                   class_weight=clweight,
                                   min_samples_leaf=minleaf, 
                                   random_state=0, 
                                  )
    
    clf=model.fit(X_in,y)
    
    pred_labels_tr = model.predict(X_in)

    print('*************** Tree Summary ***************')
    print('Classes: ', clf.classes_)
    print('Tree Depth: ', clf.tree_.max_depth)
    print('No. of features: ', clf.n_features_)
    #print('Feature Importance: ')
    #print(list(zip(X.columns, clf.feature_importances_)))
    print('--------------------------------------------------------')
    print("")
    
    print('*************** Evaluation on Training Data ***************')
    score_tr = model.score(X_in, y)
    print('Accuracy Score: ', score_tr)
    # Look at classification report to evaluate the model
    print(classification_report(y, pred_labels_tr))
    print('--------------------------------------------------------')
    
    # Return relevant data for chart plotting
    return clf


clf_pca=fitting(X_trans_pca, y, 'gini', 'best', mdepth=3, clweight=None, minleaf=10)

clf_lda = fitting(X_trans_lda, y, 'gini', 'best', mdepth=3, clweight=None, minleaf=10)
