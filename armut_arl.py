import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.use("Qt5Agg")

colors = ['#FFB6B9', '#FAE3D9', '#BBDED6', '#61C0BF', "#CCA8E9", "#F67280"]

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.expand_frame_repr", False)
"""

"""
df_ = pd.read_csv("datasets/armut_data.csv")
df = df_.copy()
df.head()
"""
   UserId  ServiceId  CategoryId           CreateDate
0   25446          4           5  2017-08-06 16:11:00
1   22948         48           5  2017-08-06 16:12:00
2   10618          0           8  2017-08-06 16:13:00
3    7256          9           4  2017-08-06 16:14:00
4   25446         48           5  2017-08-06 16:16:00

- ServiceID represents a different service for each CategoryID.
"""
df.shape  # (162523, 4)
df["UserId"].nunique()  # 24826
df["ServiceId"].nunique()  # 50
df["CategoryId"].nunique()  # 12


def check_df(dataframe, head=5, tail=5):
    print("*" * 70)
    print(" Shape ".center(70, "*"))
    print("*" * 70)
    print(dataframe.shape)

    print("*" * 70)
    print(" Types ".center(70, "*"))
    print("*" * 70)
    print(dataframe.dtypes)

    print("*" * 70)
    print(" Head ".center(70, "*"))
    print("*" * 70)
    print(dataframe.head(head))

    print("*" * 70)
    print(" Tail ".center(70, "*"))
    print("*" * 70)
    print(dataframe.tail(tail))

    print("*" * 70)
    print(" NA ".center(70, "*"))
    print("*" * 70)
    print(dataframe.isnull().sum())

    print("*" * 70)
    print(" Quantiles ".center(70, "*"))
    print("*" * 70)
    print(dataframe.describe([.01, .05, .1, .5, .9, .95, .99]).T)

    print("*" * 70)
    print(" Duplicate Rows ".center(70, "*"))
    print("*" * 70)
    print(dataframe.duplicated().sum())

    print("*" * 70)
    print(" Uniques ".center(70, "*"))
    print("*" * 70)
    print(dataframe.nunique())


check_df(df)

df["CreateDate"] = pd.to_datetime(df["CreateDate"])
df["Year"] = df["CreateDate"].dt.year
df["Month"] = df["CreateDate"].dt.month
df["Day"] = df["CreateDate"].dt.day_name()
df.head()


def cat_plots(dataframe, cat_col, save=False):
    print("".center(100, "#"))
    print(dataframe[cat_col].value_counts())

    plt.figure(figsize=(15, 10))
    plt.suptitle(cat_col.capitalize(), size=16)
    plt.subplot(1, 2, 1)
    plt.title("Percentages")
    plt.pie(dataframe[cat_col].value_counts().values.tolist(),
            labels=dataframe[cat_col].value_counts().keys().tolist(),
            labeldistance=1.1,
            wedgeprops={'linewidth': 3, 'edgecolor': 'white'},
            colors=colors,
            autopct='%1.0f%%')

    plt.subplot(1, 2, 2)
    plt.title("Countplot")
    sns.countplot(data=dataframe, x=cat_col, palette=colors)
    plt.tight_layout(pad=3)
    if save:
        plt.savefig(f"graphs/{cat_col}.png")
    plt.show(block=True)


cat_plots(df, "CategoryId", True)
cat_plots(df, "Year", True)
cat_plots(df, "Month", True)
cat_plots(df, "Day", True)

"""
The dataset consists of dates and times when service orders are placed, and there is no definition of a basket (invoice,
etc.). In order to apply Association Rule Learning, a definition of a basket (invoice, etc.) must be created. In this 
case, the basket definition is the services that each customer takes on a monthly basis. For example, the services 9_4
and 46_4 that the customer with ID 7256 took in August 2017 represent one basket; and the services 9_4 and 38_4 that 
the customer took in October 2017 represent another basket. The baskets must be uniquely identified with an ID. To do
this, we can first create a new date variable containing only the year and month.
"""

df["Service"] = df[["ServiceId", "CategoryId"]].apply(lambda x: str(x[0]) + "_" + str(x[1]), axis=1)

df["Transaction"] = df[["UserId", "Year", "Month"]].apply(lambda x: str(x[0]) + "_" + str(x[1]) + "-" + str(x[2]),
                                                          axis=1)
df.head()
"""
   UserId  ServiceId  CategoryId          CreateDate  Year  Month     Day Service   Transaction
0   25446          4           5 2017-08-06 16:11:00  2017      8  Sunday     4_5  25446_2017-8
1   22948         48           5 2017-08-06 16:12:00  2017      8  Sunday    48_5  22948_2017-8
2   10618          0           8 2017-08-06 16:13:00  2017      8  Sunday     0_8  10618_2017-8
3    7256          9           4 2017-08-06 16:14:00  2017      8  Sunday     9_4   7256_2017-8
4   25446         48           5 2017-08-06 16:16:00  2017      8  Sunday    48_5  25446_2017-8
"""
df = df.groupby(["Transaction", "Service"])["Service"].count().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)
df.head()

"""
Service        0_8  10_9  11_11  12_7  13_11  14_7  15_1  16_8  17_5  18_4  19_6  1_4  20_5  21_5  22_0  23_10  24_10  
Transaction                                                                                                                                                                                                                                                               
0_2017-8         0     0      0     0      0     0     0     0     0     0     0    0     0     0     0      0      0    
0_2017-9         0     0      0     0      0     0     0     0     0     0     0    0     0     0     0      0      0     
0_2018-1         0     0      0     0      0     0     0     0     0     0     0    0     0     0     0      0      0       
0_2018-4         0     0      0     0      0     1     0     0     0     0     0    0     0     0     0      0      0       
10000_2017-12    1     0      0     0      0     0     1     0     0     0     0    0     0     0     0      0      0      
"""
print(f"Total number of Transactions: {df.shape[0]}")
print(f"Total number of Services: {df.shape[1]}")
"""
Total number of Transactions: 71220
Total number of Services: 50
"""

frequent_itemset = apriori(df, min_support=0.01, use_colnames=True)

frequent_itemset.sort_values("support", ascending=False).head(10)
"""
     support itemsets
8   0.238121   (18_4)
19  0.130286    (2_0)
5   0.120963   (15_1)
39  0.067762   (49_1)
28  0.066568   (38_4)
3   0.056627  (13_11)
12  0.047515   (22_0)
9   0.045563   (19_6)
15  0.042895   (25_0)
7   0.041533   (17_5)
"""
rules = association_rules(frequent_itemset, metric="support", min_threshold=0.01)
rules.sort_values("lift", ascending=False).head()
"""
   antecedents consequents  antecedent support  consequent support   support  confidence      lift  leverage  conviction
10      (22_0)      (25_0)            0.047515            0.042895  0.011120    0.234043  5.456141  0.009082    1.249553
11      (25_0)      (22_0)            0.042895            0.047515  0.011120    0.259247  5.456141  0.009082    1.285834
18      (38_4)       (9_4)            0.066568            0.041393  0.010067    0.151234  3.653623  0.007312    1.129413
19       (9_4)      (38_4)            0.041393            0.066568  0.010067    0.243216  3.653623  0.007312    1.233418
4       (15_1)      (33_4)            0.120963            0.027310  0.011233    0.092861  3.400299  0.007929    1.072262
"""
df_result = rules[rules["lift"] > 2].sort_values("lift", ascending=False)
df_result
"""
   antecedents consequents  antecedent support  consequent support   support  confidence      lift  leverage  conviction
10      (22_0)      (25_0)            0.047515            0.042895  0.011120    0.234043  5.456141  0.009082    1.249553
11      (25_0)      (22_0)            0.042895            0.047515  0.011120    0.259247  5.456141  0.009082    1.285834
18      (38_4)       (9_4)            0.066568            0.041393  0.010067    0.151234  3.653623  0.007312    1.129413
19       (9_4)      (38_4)            0.041393            0.066568  0.010067    0.243216  3.653623  0.007312    1.233418
4       (15_1)      (33_4)            0.120963            0.027310  0.011233    0.092861  3.400299  0.007929    1.072262
5       (33_4)      (15_1)            0.027310            0.120963  0.011233    0.411311  3.400299  0.007929    1.493211
12       (2_0)      (22_0)            0.130286            0.047515  0.016568    0.127169  2.676409  0.010378    1.091260
13      (22_0)       (2_0)            0.047515            0.130286  0.016568    0.348700  2.676409  0.010378    1.335350
14       (2_0)      (25_0)            0.130286            0.042895  0.013437    0.103136  2.404371  0.007849    1.067168
15      (25_0)       (2_0)            0.042895            0.130286  0.013437    0.313257  2.404371  0.007849    1.266432
2        (2_0)      (15_1)            0.130286            0.120963  0.033951    0.260588  2.154278  0.018191    1.188833
3       (15_1)       (2_0)            0.120963            0.130286  0.033951    0.280673  2.154278  0.018191    1.209066
"""

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]


arl_recommender(df_result, "2_0", 1)
# ['22_0']
arl_recommender(df_result, "2_0", 4)
# ['22_0', '25_0', '15_1']

arl_recommender(df_result, "22_0", 1)
# ['25_0']