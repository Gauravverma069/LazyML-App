import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import datashader as ds
import datashader.transfer_functions as tf
from colorcet import fire
import plotly.express as px
# function to analysing EDA
def eda_analysis(df):

    target_col = st.sidebar.selectbox("Select Target Column", df.columns,index = len(df.columns)-1)
    y = df[target_col]
    X = df.drop(columns = target_col)
    num_cols = X.select_dtypes(exclude= "O").columns.tolist()
    cat_cols = X.select_dtypes(include= "O").columns.tolist()
    st.write("num_cols",tuple(num_cols))
    st.write("cat_cols",tuple(cat_cols))
    st.divider()

    results = []
    for column in X[num_cols].columns:
        skewness = X[column].skew()
        kurtosis = X[column].kurtosis()
        
        skewness_html = f'<span style="color: {"red" if abs(skewness) > .5 else "white"}">{skewness:.2f}</span>'
        kurtosis_html = f'<span style="color: {"red" if abs(kurtosis) > 3 else "white"}">{kurtosis:.2f}</span>'
        
        results.append({
            'Column': column,
            'Skewness': skewness,
            'Kurtosis': kurtosis,
            'Skewness_': skewness_html,
            'Kurtosis_': kurtosis_html
        })

    result_df = pd.DataFrame(results)

    # Display the data types of Skewness and Kurtosis columns
    # st.write("Data types of Skewness and Kurtosis columns:", result_df[["Skewness", "Kurtosis"]].dtypes)

    if st.toggle("Show Skewness and Kurtosis of DataFrame columns"):
        st.write("Columns with Skewness and Kurtosis:")
        if st.checkbox("Filter Skewed columns"):
            filtered_df = result_df[abs(result_df["Skewness"]) > 0.5]
            st.write(filtered_df[['Column', 'Skewness_', 'Kurtosis_']].to_html(escape=False), unsafe_allow_html=True)
        else:
            st.write(result_df[['Column', 'Skewness_', 'Kurtosis_']].to_html(escape=False), unsafe_allow_html=True)

    st.divider()
    st.write("Plotting Numerical Columns for Visual EDA")

     # Create two columns
    column1, column2 = st.columns(2)

    # Checkbox for plotting distribution in the first column
    with column1:
        plot_distribution = st.checkbox("Plot Distribution of Target Column")

    # Show the second checkbox in the second column only if the first checkbox is clicked
    if plot_distribution:
        with column2:
            show_kde = st.checkbox("Show KDE Plot")
        kde = show_kde
    else:
        kde = False

    # Plot the histogram if the first checkbox is checked
    if plot_distribution:
        fig, ax = plt.subplots()
        sns.histplot(y, ax=ax, kde=kde)

        # Show the plot in the Streamlit app
        st.pyplot(fig)
    
    column3, column4 = st.columns(2)
    with column3:
        plot_distribution_nc =st.checkbox("Plot Distribution of Input Numerical columns")
    if plot_distribution_nc:
        with column4:
            show_kde_1 = st.checkbox("Show KDE Plot for Numerical Columns")
        kde_1 = show_kde_1
    if plot_distribution_nc:
        for column in num_cols:
            fig, ax = plt.subplots()
            sns.histplot(df[column], ax=ax, kde=kde_1)
            st.write(f"Distribution of {column}:")
            st.pyplot(fig)
    st.divider()
    # plot count plot for categorical columns
    st.write("Plotting Categorical Columns for Visual EDA")
    if st.checkbox("Plot Distribution of Input Categorical columns") :
        for column in cat_cols:
            fig, ax = plt.subplots()
            fig = px.histogram(df.fillna('Null'), x=column, color=target_col)
            st.write(fig)
    
    st.divider()
    # plot correlation matrics using plotly
    st.write("Plotting Correlation Matrix for Numerical Columns")

    column5, column6 = st.columns(2)
    with column5:
        plot_distribution =st.checkbox("Plot Correlation Matrix")
    if plot_distribution:
        with column6:
            show_value = st.checkbox("Correlation values > 0.5")
        if show_value:
            # Compute correlation matrix
            corr_matrix = df[num_cols].corr()

            # Plot correlation matrix heatmap
            fig = px.imshow(corr_matrix[abs(corr_matrix)>0.5], color_continuous_scale='RdBu')

            # Add annotations for values greater than 0.5
            for i in range(corr_matrix.shape[0]):
                for j in range(corr_matrix.shape[1]):
                    correlation_value = corr_matrix.iloc[i, j]
                    if abs(correlation_value) > 0.5:  # Filter values greater than 0.5
                        fig.add_annotation(
                            x=i, y=j,
                            text=str(round(correlation_value, 2)),
                            showarrow=False
                        )

            # Update layout
            fig.update_layout(
                xaxis=dict(side="top"),
                width=600,
                height=600,
                margin=dict(l=20, r=20, t=40, b=20)
            )

            # Display the heatmap
            st.write(fig)
    if plot_distribution and not show_value:

   
        corr_matrix = df[num_cols].corr()
        fig = px.imshow(corr_matrix, color_continuous_scale='RdBu')
        for i in range(corr_matrix.shape[0]):
            for j in range(corr_matrix.shape[1]):
                fig.add_annotation(
                    x=i, y=j,
                    text=str(round(corr_matrix.iloc[i, j], 2)),
                    showarrow=False
                )

        # Update the layout to ensure annotations are displayed properly
        fig.update_layout(
            xaxis=dict(side="top"),
            width=600,
            height=600,
            margin=dict(l=20, r=20, t=40, b=20)
        )

        st.write(fig)
    st.divider()
    outlier_cols = st.multiselect("Select Continous numerical columns for Outlier Plot",num_cols)

    # plot px.boxplot for outlier cols
    if st.toggle("Toggle for Violin Plot"):
        if st.checkbox("Plot BoxPlot for Outlier Cols"):
            if st.toggle("Split by Target"):
                for col in outlier_cols:
                    fig = px.violin(df, x=col,color=y)
                    st.write(fig)
                st.divider()
            else:
                for col in outlier_cols:
                    fig = px.violin(df, x=col)
                    st.write(fig)
                st.divider()
        if st.checkbox("check outlier distribution of Target column"):
            fig = px.violin(y)
            st.write(fig)

    else:
        if st.checkbox("Plot BoxPlot for Outlier Cols"):
            if st.toggle("Split by Target"):
                for col in outlier_cols:
                    fig = px.box(df, x=col,color=y)
                    st.write(fig)
                st.divider()
            else:
                for col in outlier_cols:
                    fig = px.box(df, x=col)
                    st.write(fig)
                st.divider()
        if st.checkbox("check outlier distribution of Target column"):
            fig = px.box(y)
            st.write(fig)


    # plot scatter plot using px
    st.divider()
    
    if st.checkbox("Plot Scatter Plot"):
        column7, column8,column9 = st.columns(3)
        with column7:
        
        
            # Select y-axis column
            y_col = st.selectbox("Select y axis column", df.columns)
            
            # Filter categorical columns for the x-axis selection
            categorical_columns = df.columns
        with column8:    
            # Allow user to select the x-axis column from categorical columns
            x_col = st.selectbox("Select x axis column", categorical_columns)
        with column9:
            hue_col = st.selectbox("Select Hue column",categorical_columns)   
            # Plot scatter plot using Plotly
        fig = px.scatter(df, x=x_col, y=y_col, color=hue_col)
        st.write(fig)
        
    # barchart and line chart
    st.divider()
    if st.checkbox("Plot Bar Chart"):
        column10, column11 = st.columns(2)
        with column10:
            # Select y-axis column
            y_col = st.selectbox("Select y axis column", df.columns)
            
            # Filter categorical columns for the x-axis selection
            categorical_columns = df.columns
        with column11:
            # Allow user to select the x-axis column from categorical columns
            x_col = st.selectbox("Select x axis column", categorical_columns)
        fig = px.bar(df, x=x_col, y=y_col,color = x_col)
        st.write(fig)
    st.divider()
    if st.checkbox("Plot Line Chart"):
        column12, column13,colx = st.columns(3)
        with column12:
            # Select y-axis column
            y_col = st.selectbox("Select y axis column", df.columns)
            
            # Filter categorical columns for the x-axis selection
            categorical_columns = df.columns
        with column13:
            # Allow user to select the x-axis column from categorical columns
            x_col = st.selectbox("Select x axis column", categorical_columns)
        with colx:
            hue_col1 = st.selectbox("Select Line split column",categorical_columns)
        fig = px.line(df.sort_values(by = y_col), x=x_col, y=y_col,color = hue_col1)
        st.write(fig)
    st.divider()
    # plot pie chart
    if st.checkbox("Plot Pie Chart "):
        column14, column15 = st.columns(2)
        with column14:
            # Select y-axis column
            y_col = st.selectbox("Select values columns", df.columns)
            
            # Filter categorical columns for the x-axis selection
            categorical_columns = df.columns
        with column15:
            # Allow user to select the x-axis column from categorical columns
            x_col = st.selectbox("Select names column", categorical_columns)
        fig = px.pie(df, values=y_col, names=x_col)
        st.write(fig)
    
    st.divider()
    # check if there are latitude and longitude columns
    if st.checkbox("Plot on Map"):
        lat_col = st.selectbox("Select Latitute Column",df.columns)
        long_col = st.selectbox("Select Longitude Column",df.columns)
        color = st.selectbox

        # # Create the datashader canvas and aggregate points
        # cvs = ds.Canvas(plot_width=1000, plot_height=1000)
        # agg = cvs.points(df, x=long_col, y=lat_col)

        # # Get the coordinates for the mapbox layer
        # coords_lat, coords_lon = agg.coords[lat_col].values, agg.coords[long_col].values
        # coordinates = [
        #     [coords_lon[0], coords_lat[0]],
        #     [coords_lon[-1], coords_lat[0]],
        #     [coords_lon[-1], coords_lat[-1]],
        #     [coords_lon[0], coords_lat[-1]]
        # ]

        # # Generate the datashader image
        # img = tf.shade(agg, cmap=fire)[::-1].to_pil()

        # # Create the Plotly figure with a mapbox layer
        # fig = px.scatter_mapbox(df[:1], lat=lat_col, lon=long_col, zoom=10)  # Adjust zoom level as needed
        # fig.update_layout(mapbox_style="carto-darkmatter",
        #                 mapbox_layers=[
        #                     {
        #                         "sourcetype": "image",
        #                         "source": img,
        #                         "coordinates": coordinates
        #                     }
        #                 ])

        # # Display the figure in Streamlit
        # st.plotly_chart(fig)

        # Create a scatter mapbox plot with vibrant colors and custom marker sizes
        if st.button("Proceed to plot map"):
            fig = px.scatter_mapbox(df, lat=lat_col, lon=long_col, 
                                    
                                    size_max=15,  # Max marker size
                                    mapbox_style="open-street-map",  # Using a different map style for vibrancy
                                    zoom=1,
                                    title='Latitude and Longitude Plotting')

            # Customize the layout for more vibrant appearance
            fig.update_layout(mapbox_accesstoken='your_mapbox_access_token') 
            st.write(fig)

    



