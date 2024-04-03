# cc_fraud_detection




```
              cc_num         amt         zip         lat        long  \
count  1.296675e+06  1296675.00  1296675.00  1296675.00  1296675.00   
mean   4.171920e+17       70.35    48800.67       38.54      -90.23   
std    1.308806e+18      160.32    26893.22        5.08       13.76   
min    6.041621e+10        1.00     1257.00       20.03     -165.67   
25%    1.800429e+14        9.65    26237.00       34.62      -96.80   
50%    3.521417e+15       47.52    48174.00       39.35      -87.48   
75%    4.642255e+15       83.14    72042.00       41.94      -80.16   
max    4.992346e+18    28948.90    99783.00       66.69      -67.95   

         city_pop     unix_time   merch_lat  merch_long    is_fraud  
count  1296675.00  1.296675e+06  1296675.00  1296675.00  1296675.00  
mean     88824.44  1.349244e+09       38.54      -90.23        0.01  
std     301956.36  1.284128e+07        5.11       13.77        0.08  
min         23.00  1.325376e+09       19.03     -166.67        0.00  
25%        743.00  1.338751e+09       34.73      -96.90        0.00  
50%       2456.00  1.349250e+09       39.37      -87.44        0.00  
75%      20328.00  1.359385e+09       41.96      -80.24        0.00  
max    2906700.00  1.371817e+09       67.51      -66.95        1.00  


        trans_date_trans_time           merchant       category        first  \
count                1296675            1296675        1296675      1296675   
unique               1274791                693             14          352   
top      2019-04-22 16:02:01  fraud_Kilback LLC  gas_transport  Christopher   
freq                       4               4403         131659        26669   

           last   gender                      street        city    state  \
count   1296675  1296675                     1296675     1296675  1296675   
unique      481        2                         983         894       51   
top       Smith        F  0069 Robin Brooks Apt. 695  Birmingham       TX   
freq      28794   709863                        3123        5617    94876   

                      job         dob                         trans_num  
count             1296675     1296675                           1296675  
unique                494         968                           1296675  
top     Film/video editor  1977-03-23  0b242abb623afc578575680df30655b9  
freq                 9779        5636                                 1

amt statistics for fraudulent transactions :  count    7506.000000
mean      531.320092
std       390.560070
min         1.060000
25%       245.662500
50%       396.505000
75%       900.875000
max      1376.040000
Name: amt, dtype: float64

amt statistics for legitimate transactions :  count    1.289169e+06
mean     6.766711e+01
std      1.540080e+02
min      1.000000e+00
25%      9.610000e+00
50%      4.728000e+01
75%      8.254000e+01
max      2.894890e+04
Name: amt, dtype: float64

```


![image](https://github.com/BillysKes/cc_fraud_detection/assets/73298709/cc73219e-c675-4607-9d7a-4600fbbd7ea6)


![image](https://github.com/BillysKes/cc_fraud_detection/assets/73298709/d81fb83a-87eb-4d88-9628-e0b312793435)

![image](https://github.com/BillysKes/cc_fraud_detection/assets/73298709/8786de1f-4b86-40ae-b5a5-a28234d52444)

![image](https://github.com/BillysKes/cc_fraud_detection/assets/73298709/e5b530a8-6413-4df9-a3de-7bc184a6dbfa)


![image](https://github.com/BillysKes/cc_fraud_detection/assets/73298709/582074ad-4d77-4f38-b578-3998bed06f26)

The high majority of cardholders spent less than 500$ per transaction.

![image](https://github.com/BillysKes/cc_fraud_detection/assets/73298709/0b210192-3b6d-44f1-84c8-4df36c2a28e8)


![image](https://github.com/BillysKes/cc_fraud_detection/assets/73298709/9707793f-e494-4975-b102-76cf506f7723)

![image](https://github.com/BillysKes/cc_fraud_detection/assets/73298709/4dc7ed96-6ea4-4624-955f-a3216fcf954c)

![image](https://github.com/BillysKes/cc_fraud_detection/assets/73298709/1012c594-1270-44de-9b47-2b5d41157e76)


![image](https://github.com/BillysKes/cc_fraud_detection/assets/73298709/d9c1abf3-9d8d-44db-bea9-c6fce9f74c61)

![image](https://github.com/BillysKes/cc_fraud_detection/assets/73298709/ee7a10a6-b76b-48f8-b95d-2e2ef968d224)

![image](https://github.com/BillysKes/cc_fraud_detection/assets/73298709/0b15096b-fc7b-4ba2-9282-85e8bb8b8f75)

![image](https://github.com/BillysKes/cc_fraud_detection/assets/73298709/c440f2bd-3cb9-49be-aa80-688b25bf88d4)

