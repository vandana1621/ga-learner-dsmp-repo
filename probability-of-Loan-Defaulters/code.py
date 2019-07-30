# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# code starts here


df = pd.read_csv(path)
#print(df)

p_a =len(df[df['fico']>700])/len(df)
print("p_a: ", p_a)

p_b =len(df[df['purpose']=='debt_consolidation'])/len(df)
print("p_b: ", p_b)

df1 = df[df['purpose']=='debt_consolidation']
#print(df1)

#p_a_b = len(df[(df['purpose']=='debt_consolidation') & (df['fico']>700)])/len(df)
p_b_a = len(df1[df1['fico']>700])/len(df)
print("p_b_a: ", p_b_a)

p_a_b =(p_b_a * p_a) / p_b
print("p_a_b: ", p_a_b)

result= (p_a_b == p_a)
print(result)

# code ends here


# --------------
# code starts here


prob_lp = len(df[df['paid.back.loan']=='Yes'])/len(df)
print("prob_lp: ", prob_lp)

prob_cs = len(df[df['credit.policy']=='Yes'])/len(df)
print("prob_cs: ", prob_cs)

new_df = df[df['paid.back.loan']=='Yes']
prob_pd_cs = len(new_df[new_df['credit.policy']=='Yes'])/len(new_df)
print("prob_pd_cs: ",prob_pd_cs)
bayes =(prob_pd_cs*prob_lp)/prob_cs
print("bayes: ", bayes)


# code ends here


# --------------
# code starts here

purpose_count = df['purpose'].value_counts()
#print(purpose_count)
purpose_count.plot.bar()
plt.show()

df1=df[df['paid.back.loan']=='No']
#print(df1)

purpose_count1 = df1['purpose'].value_counts()
#print(purpose_count1)
purpose_count1.plot.bar()
plt.show()

# code ends here


# --------------
# code starts here

inst_median = df['installment'].median()
print("inst_median: ",inst_median)

inst_mean = df['installment'].mean()
print("inst_mean: ", inst_mean)

plt.hist(df['installment'])
plt.title('installment')
plt.show()

plt.hist(df['log.annual.inc'])
plt.title('Annual Income')
plt.show()


# code ends here


