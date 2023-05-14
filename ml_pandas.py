# Pandas DataFrame is two-dimensional size-mutable, potentially heterogeneous
# tabular data structure with labeled axes (rows and columns). 

import pandas as pd
import numpy as np

# # list of strings
# lst = ['Geeks', 'For', 'Geeks', 'is', 'portal', 'for', 'Geeks']
# # Calling DataFrame constructor on list
# df = pd.DataFrame(lst)
# print(df)

# # intialise data of lists.
# data = {'Name':['Tom', 'nick', 'krish', 'jack'],
#         'Age':[20, 21, 19, 18]}
# # Create DataFrame
# df = pd.DataFrame(data)
# # Print the output.
# print(df)

# # Define a dictionary containing employee data
# data = {'Name':['Jai', 'Princi', 'Gaurav', 'Anuj'],
#         'Age':[27, 24, 22, 32],
#         'Address':['Delhi', 'Kanpur', 'Allahabad', 'Kannauj'],
#         'Qualification':['Msc', 'MA', 'MCA', 'Phd']}
# # Convert the dictionary into DataFrame 
# df = pd.DataFrame(data)
# print(df)
# # select two columns
# print(df[['Name', 'Qualification']])


# # making data frame from csv file
# data = pd.read_csv("nba.csv", index_col ="Name")
# print(data)
# # retrieving row by loc method
# first = data.loc["Avery Bradley"]
# first = data.iloc[:,:]
# second = data.loc["R.J. Hunter"]
# print(first, "\n\n\n", second)


# dictionary of lists
# dict = {'Name':['utkarsh', 'samarth', 'khushi', 'shreya'],
#         'First Score':[100, 90, np.nan, 95],
#         'Second Score': [30, 45, 56, np.nan],
#         'Third Score':[np.nan, 40, 80, 98]}
# # creating a dataframe from list
# df = pd.DataFrame(dict)
# print(df)
# using isnull() function  
# print(df.isnull()) 
# print(df.fillna(0))
# print(df.dropna())
# print(df)
# iterating over rows using iterrows() function 
# for i, j in df.iterrows():
#     print(i, j)
#     print()

'''
FUNCTION	        DESCRIPTION

index()	            Method returns index (row labels) of the DataFrame
insert()	        Method inserts a column into a DataFrame
add()	            Method returns addition of dataframe and other, element-wise (binary operator add)
sub()	            Method returns subtraction of dataframe and other, element-wise (binary operator sub)
mul()	            Method returns multiplication of dataframe and other, element-wise (binary operator mul)
div()	            Method returns floating division of dataframe and other, element-wise (binary operator truediv)
unique()	        Method extracts the unique values in the dataframe
nunique()	        Method returns count of the unique values in the dataframe
value_counts()	    Method counts the number of times each unique value occurs within the Series
columns()	        Method returns the column labels of the DataFrame
axes()	            Method returns a list representing the axes of the DataFrame
isnull()	        Method creates a Boolean Series for extracting rows with null values
notnull()	        Method creates a Boolean Series for extracting rows with non-null values
between()	        Method extracts rows where a column value falls in between a predefined range
isin()	            Method extracts rows from a DataFrame where a column value exists in a predefined collection
dtypes()	        Method returns a Series with the data type of each column. The result’s index is the original DataFrame’s columns
astype()	        Method converts the data types in a Series
values()	        Method returns a Numpy representation of the DataFrame i.e. only the values in the DataFrame will be returned, the axes labels will be removed
sort_values()-     
Set1, Set2	        Method sorts a data frame in Ascending or Descending order of passed Column
sort_index()	    Method sorts the values in a DataFrame based on their index positions or labels instead of their values but sometimes a data frame is made out of two or more data frames and hence later index can be changed using this method
loc[]	            Method retrieves rows based on index label
iloc[]	            Method retrieves rows based on index position
ix[]	            Method retrieves DataFrame rows based on either index label or index position. This method combines the best features of the .loc[] and .iloc[] methods
rename()	        Method is called on a DataFrame to change the names of the index labels or column names
columns()	        Method is an alternative attribute to change the coloumn name
drop()	            Method is used to delete rows or columns from a DataFrame
pop()	            Method is used to delete rows or columns from a DataFrame
sample()	        Method pulls out a random sample of rows or columns from a DataFrame
nsmallest()	        Method pulls out the rows with the smallest values in a column
nlargest()	        Method pulls out the rows with the largest values in a column
shape()	            Method returns a tuple representing the dimensionality of the DataFrame
ndim()	            Method returns an int representing the number of axes / array dimensions.
                    Returns 1 if Series, otherwise returns 2 if DataFrame
dropna()	        Method allows the user to analyze and drop Rows/Columns with Null values in different ways
fillna()	        Method manages and let the user replace NaN values with some value of their own
rank()	            Values in a Series can be ranked in order with this method
query()	            Method is an alternate string-based syntax for extracting a subset from a DataFrame
copy()	            Method creates an independent copy of a pandas object
duplicated()	    Method creates a Boolean Series and uses it to extract rows that have duplicate values
drop_duplicates()	Method is an alternative option to identifying duplicate rows and removing them through filtering
set_index()	        Method sets the DataFrame index (row labels) using one or more existing columns
reset_index()	    Method resets index of a Data Frame. This method sets a list of integer ranging from 0 to length of data as index
where()	            Method is used to check a Data Frame for one or more condition and return the result accordingly. By default, the rows not satisfying the condition are filled with NaN value
'''

df=pd.DataFrame(np.arange(20).reshape(5,4),index=["row1","row2","row3","row4","row5"],columns=["col1","col2","col3","col4"])
# print(df)
# df.to_csv("test_case.csv")
print(df.iloc[:3,:2].values)
















