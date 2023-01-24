import streamlit as st

import seaborn as sns
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import joblib
import math

from sklearn.metrics import r2_score            # 決定係数
from sklearn.metrics import mean_squared_error  # RMSE
from sklearn.decomposition import PCA

import math
from scipy.stats import norm
from scipy import stats
import io

st.set_page_config(layout="wide")
st.sidebar.title('ツール選択')

bunrui = ["重回帰分析"]#,"主成分分析","要約統計量"
b1 = st.sidebar.selectbox("分類",bunrui)

if b1=="重回帰分析":

    a1 = st.sidebar.number_input('説明変数',value=6)
    a2 = st.sidebar.number_input('目的変数',value=1)

    sns.set(font='Meiryo')

    st.title("インプット情報")
    data = st.text_area('データ入力欄')

    if st.sidebar.button('実行'):

        #sns.set(font_scale = 2)
        #plt.style.use('default')
        #plt.rcParams['xtick.direction'] = 'in'
        #plt.rcParams['ytick.direction'] = 'in'
        
        data = pd.read_csv(io.StringIO(data), header=0,sep='\s+')
        st.write(data)

        st.title("重回帰分析結果")
        #csvファイルの呼び出し
        #data = pd.read_clipboard()
        #回帰モデルの呼び出し
        clf = linear_model.LinearRegression()
        #説明変数
        X = data.iloc[:,0:a1].values
        
        #目的変数
        Y = data.iloc[:,a1].values
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=0)

        # 予測モデルを作成（重回帰）
        clf.fit(X_train, Y_train)

        # 回帰係数と切片の抽出
        a = clf.coef_
        b = clf.intercept_  

        cola,colb,colc = st.columns(3)

        with cola:
            st.subheader("説明変数入力内容")
            st.dataframe(X,400,200)

        with colb:
            st.subheader("目的変数入力内容")
            st.dataframe(Y,400,200)

        with colc:
            st.subheader("回帰係数算出結果")
            # 回帰係数
            st.dataframe(a,400,200)
            st.write("切片：",b)
            st.write("決定係数",clf.score(X_train, Y_train))

        col1,col2 = st.columns(2)

        with col1:
            st.subheader("相関係数カラーマップ")
            sns.set(rc = {'figure.figsize':(8,4)})
            sns.set(font='Meiryo')
            g2 = sns.heatmap(data.corr(),cmap='YlGnBu',annot=True)
            st.pyplot(g2.figure)
            st.code("【相関係数目安】\n0~0.3未満：ほぼ無関係\n0.3~0.5未満：非常に弱い相関\n0.5~0.7未満：相関がある\n0.7~0.9未満：強い相関\n0.9以上：非常に強い相関")

        with col2:
            st.subheader("回帰係数比較グラフ")
            plt.clf()
            plt.close()
            x1 = pd.DataFrame(data.columns)
            x2 = x1[0:a1]
            x3 = x2.iloc[:, 0]
            x=[1,2,3,4,5,6]
            sns.set(font='Meiryo')
            g3 = sns.barplot(a,x3) 
            st.pyplot(g3.figure)

        st.subheader("行列散布図(ペアプロット)")
        g1 = sns.pairplot(data)
        sns.set(font='Meiryo')
        st.pyplot(g1.figure)

elif b1=="主成分分析":
    
    if st.sidebar.button('実行'):
        st.title("主成分分析結果")
    
        #csvファイルの呼び出し
        data = pd.read_clipboard()

        # 変数の標準化
        df_workers_std = data.apply(lambda x: (x-x.mean())/x.std(), axis=0)

        # 主成分分析の実行
        pca = PCA()
        pca.fit(df_workers_std)

        # データを主成分に変換
        pca_row = pca.transform(df_workers_std)

        # 寄与率を求める
        pca_col = ["PC{}".format(x + 1) for x in range(len(df_workers_std.columns))]
        df_con_ratio = pd.DataFrame([pca.explained_variance_ratio_], columns = pca_col)
        kiyodo=df_con_ratio.transpose()
        kiyodo.columns = ["寄与率"]

        # 累積寄与率を図示する
        cum_con_ratio = np.hstack([0, pca.explained_variance_ratio_]).cumsum()
        cum_con_ratio = pd.DataFrame(cum_con_ratio)
        cum_con_ratio.columns = ["PC"]
        cum_con_ratio['No'] = range(1, len(cum_con_ratio.index) + 1)
        x=cum_con_ratio["No"]
        y=cum_con_ratio["PC"]

        
        cola,colb,colc = st.columns([4,1,3])
        with cola:
            st.subheader("累積寄与率")
            fig, ax = plt.subplots()
            ax.scatter(x,y,alpha=0.85)
            ax.plot(x,y,alpha=0.85)
            st.pyplot(fig)

        with colb:
            st.subheader("寄与率")
            # 主成分負荷量を求める
            df_pca = pd.DataFrame(pca_row, columns = pca_col)
            df_pca_vec = pd.DataFrame(pca.components_, columns=data.columns,
                                    index=["PC{}".format(x + 1) for x in range(len(df_pca.columns))])

            st.table(kiyodo)

        with colc:
            st.subheader("主成分負荷量")
            st.table(df_pca_vec)

        fig1, ax1 = plt.subplots()
        
        for x, y, name in zip(pca.components_[0], pca.components_[1], data.columns[0:]):
            plt.text(x, y, name)
        ax1.scatter(pca.components_[0], pca.components_[1])
        st.pyplot(fig1)


        



elif b1=="要約統計量":
    a0 = st.sidebar.text_input('グラフタイトル',value="Title")
    a1 = st.sidebar.number_input('規格上限',value=2)
    a2 = st.sidebar.number_input('規格下限',value=1)

    if st.sidebar.button('実行'):
            st.title("要約統計量出力結果")
            #データの呼び出し
            df = pd.read_clipboard()

            cola,colb,colc = st.columns(3)
            #plt.style.use('default')

            with cola:
                st.subheader("ヒストグラム")
                fig, ax = plt.subplots()
                #ax.hist(df,color = CL1,alpha=0.85)
                ax.hist(df,alpha=0.85)
                plt.ylabel("サンプル数",fontsize=10,fontname="Meiryo")
                plt.title(a0,fontsize=12,fontname="Meiryo")
                plt.tick_params(labelsize=10)
                fig.set_figheight(5/2.5)
                fig.set_figwidth(8/2.5)
                st.pyplot(fig)
            with colb:
                st.subheader("カーネル密度推定")
                #fig1, ax = plt.subplots()
                #ax.kde(df)
                fig1 = df.plot.kde(figsize=(8/2.5,5.25/2.5),linewidth=1)
                plt.ylabel("Density",fontsize=10,fontname="Meiryo")
                plt.title(a0,fontsize=12,fontname="Meiryo")
                plt.tick_params(labelsize=10)
                plt.legend(prop={'size':10}, loc='best')
                #fig1.set_figheight(4)
                st.pyplot(fig1.figure)

            with colc:
                df1= df.iloc[:,0]
                ave = np.average(df1)
                nor = np.median(df1)
                std = np.std(df1)
                min = df1.min()
                max = df1.max()
                smin = ave-std*4
                smax = ave+std*4
                cp = (a1-a2)/6/std
                cpk1 = (ave-a2)/3/std
                cpk2 = (a1-ave)/3/std
                count = df1.count()
                bmax=norm.pdf(ave,ave,std)

                st.subheader("正規分布化")
                sX = np.arange(smin-2*std,smax+2*std,(a1-a2)/100)
                sY = norm.pdf(sX,ave,std)
                fig2, ax2 = plt.subplots()
                plt.ylabel("Density",fontsize=10,fontname="Meiryo")
                plt.title(a0,fontsize=12,fontname="Meiryo")
                plt.plot([a1,a1],[0,bmax],color="gray", linestyle = "dashed",label="規格",linewidth=0.75)
                plt.plot([a2,a2],[0,bmax],color="gray", linestyle = "dashed",linewidth=0.75)
                plt.plot([smax,smax],[0,0.1],color="red",label="4シグマ",linewidth=0.75)
                plt.plot([smin,smin],[0,0.1],color="red",linewidth=0.75)
                plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.5, fontsize=10)
                ax2.plot(sX,sY,linewidth=1.0)
                plt.tick_params(labelsize=10)
                fig2.set_figheight(5.3/2.5)
                fig2.set_figwidth(8/2.5)
                st.pyplot(fig2)

            st.subheader("要約統計量 算出結果")
            matome = pd.DataFrame(
                data={
                    '項目' :['データ数','平均値','中央値','標準偏差','最小値','最大値','Cp(両側)','Cpk(下限側)','Cpk(上限側)','平均値-4シグマ','平均値+4シグマ'],
                    '結果' :[count,np.round(ave,decimals=2),np.round(nor,decimals=2),np.round(std,decimals=2),np.round(min,decimals=2)
                        ,np.round(max,decimals=2),np.round(cp,decimals=2)
                        ,np.round(cpk1,decimals=2),np.round(cpk2,decimals=2),np.round(smin,decimals=2),np.round(smax,decimals=2)],
                })
            matome = matome.transpose()
            st.table(matome)
            st.caption("  正規分布化・要約統計量算出結果については最左列データのみ算出")

            cold,cole,colf = st.columns(3)

            with cold:
                               
                st.subheader("箱ひげ図")
                plt.clf()
                plt.close()
                fig3, ax3 = plt.subplots()
                ax3.boxplot(df,capprops=dict(linewidth=0.75), whiskerprops=dict(linewidth=0.75), boxprops=dict(linewidth=0.75), flierprops=dict(linewidth=0.75),medianprops=dict(color='r'))
                plt.tick_params(labelsize=10)
                plt.title(a0,fontsize=10,fontname="Meiryo")
                fig3.set_figheight(5.3/2.5)
                fig3.set_figwidth(8/2.5)
                st.pyplot(fig3)

            with cole:
                st.subheader("バイオリンプロット")
                fig4,ax4 = plt.subplots()
                violin = ax4.violinplot(df)
                for vp in violin['bodies']:
                    vp.set_linewidth(0.1)
                plt.tick_params(labelsize=10)
                plt.title(a0,fontsize=10,fontname="Meiryo")
                fig4.set_figheight(5.3/2.5)
                fig4.set_figwidth(8/2.5)
                st.pyplot(fig4)

            with colf:
                st.subheader("ジッタープロット")
                
                fig5, ax5 = plt.subplots(figsize=(8/2.5,5.25/2.5))
                plt.title(a0,fontsize=12,fontname="Meiryo")
                plt.tick_params(labelsize=10)
                p = sns.stripplot(data=df,size=3, jitter=.1, ax=ax5)
                st.pyplot(fig5.figure)
        
            st.subheader("五数要約")

            median = np.median(df1)
            upper_quartile = np.percentile(df1, 75)
            lower_quartile = np.percentile(df1, 25)

            iqr = upper_quartile - lower_quartile
            upper_whisker = df1[df1<=upper_quartile+1.5*iqr].max()
            lower_whisker = df1[df1>=lower_quartile-1.5*iqr].min()

            matome = pd.DataFrame(
                data={
                    '項目' :['中央値(Q2)','第1四分位数(Q1)','第3四分位数(Q3)','四分位範囲(IQR)','最大値','最小値'],
                    '結果' :[np.round(median,decimals=2),np.round(upper_quartile,decimals=2),np.round(lower_quartile,decimals=2),
                            np.round(iqr,decimals=2),np.round(upper_whisker,decimals=2),np.round(lower_whisker,decimals=2)],
                })
            
            matome = matome.transpose()
            st.table(matome)
            st.caption("  五数要約については最左列データのみ算出")
                