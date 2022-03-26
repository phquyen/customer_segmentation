from distutils.command.upload import upload
from email.mime import image
import numpy as np
import pandas as pd
#import ujson as json
import pickle
import streamlit as st
import matplotlib.pyplot as plt
from time import time
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler


data = pd.read_csv("df_scale.csv", encoding ='unicode_escape')

with open('gmmModel.pkl', 'rb') as file:
     gmm_model = pickle.load(file)



# GUI
menu = ['Tổng quan','Thông tin dữ liệu - EDA','Xây dựng mô hình: RFM','Xây dựng mô hình: GMM', 'Dự đoán']
choice = st.sidebar.selectbox('Menu',menu)  
if choice == 'Tổng quan':
     st.title('DATA SCIENCE PROJECT')  
     st.write('## CUSTOMER SEGMENTATION')
     st.image('image/p1.jpg')
     st.subheader('TỔNG QUAN')
     st.write('### 1. Giới thiệu Đề tài')
     st.write("""###### Custormer Segmentation - Phân khúc khách hàng là quá trình phân chia khách hàng thành các nhóm dựa trên các đặc điểm khách hàng mục tiêu chung để các công ty có thể tiếp thị cho từng nhóm một cách hiệu quả và phù hợp. Đây là bước khá quan trọng, là yếu tố giúp tăng tỉ lệ chuyển đổi cho doanh nghiệp. Nếu doanh nghiệp thực hiện tốt bước này có thể giúp doanh nghiệp phân chia ngân sách quảng cáo tốt hơn và tiết kiệm được nhiều hơn. """)
     st.image('image/cus_p1_head.jpg')

     st.write('### 2. Cách thức triển khai')
     st.write(""" - Xác định vấn đề """)
     st.write(""" - Tìm hiểu dữ liệu """)
     st.write(""" - Xử lý dữ liệu """)
     st.write(""" - Xây dựng mô hình """)
     st.write(""" - Đánh giá mô hình """)
     st.write(""" - Lựa chọn mô hình """)
     st.write(""" - Deployment & Feedback/ Act """)

     st.write('### 3. Tác giả')
     st.write(""" - Phạm Thị Thúy Quyên """)
     st.write(""" - Tôn Nữ Khánh Quỳnh """)

elif choice == 'Thông tin dữ liệu - EDA':
     st.subheader('Thông tin dữ liệu - EDA')
     st.write(""" Dữ liệu ban đầu: """)
     with st.expander("Thông tin dữ liệu:"):
          st.write("""
          - InvoiceNo: Mã định danh đơn hàng (Một số nguyên gồm 6 chữ số được chỉ định duy nhất cho mỗi đơn hàng). Nếu mã này bắt đầu bằng ký tự 'c', nó thể hiện cho các đơn hàng bị hủy
          - StockCode: Mã sản phẩm. Một số nguyên gồm 5 chữ số được chỉ định duy nhất cho mỗi sản phẩm riêng biệt
          - Description: Mô tả sản phẩm
          - Quantity: Số lượng sản phẩm cho mỗi đơn hàng
          - InvoiceDate: Ngày và thời gian mỗi đơn hàng được tạo ra
          - UnitPrice: Đơn giá
          - CustomerID: Mã định danh khách hàng. Một số nguyên gồm 5 chữ số được chỉ định duy nhất cho mỗi khách hàng
          - Country: Tên quốc gia nơi mỗi khách hàng cư trú
          """)
     st.image('image/OnlineRetail_p2_head.png')
     #st.code(f'Dữ liệu ban đầu có {data.shape[0]} dòng và {data.shape[1]} cột')

     st.write(""" Dữ liệu xử lý: """)
     st.image('image/EDA_p2_head.png')
     
     st.write(""" Chia dữ liệu thành 2 phần:""")
     st.write(""" - TotalMoney > 0: đơn hàng thành công""")
     st.write(""" - TotalMoney < 0: đơn hoàn""")

     st.write(""" Tính số chi tiêu của mỗi khách hàng""")
     st.write(""" Money = Tổng tiền (đơn thành công) - Tổng tiền (đơn hoàn)""")

     st.image('image/concat_p2_head.png')

     st.write(""" Chia dữ liệu thành 3 phần dựa vào cột 'Money'""")
     st.write(""" - Money < 0: khách hàng mang lại lợi nhuận âm""")
     st.write(""" - Money = 0: khách hàng không mang lại lợi nhuận""")
     st.write(""" - Money > 0: khách hàng mang lại lợi nhuận dương""")

     st.write(""" Sau khi xử lý dữ liệu, những cột được sử dụng để xây dựng mô hình:""")
     st.write(""" - InvoiceNo, InvoiceDate, CustomerID, Money """)
     st.image('image/EDA_final_head.png')

     st.write(""" Tần suất của các giá trị""")
     st.image('image/recency_p2.jpg')
     st.image('image/frequency_p2.jpg')
     st.image('image/monetary_p2.jpg')

elif choice == 'Xây dựng mô hình: RFM':
     st.subheader('Xây dựng mô hình: RFM')
     st.image('image/rfm_p3.jpg')
     with st.expander('Thông tin nhóm khách hàng:'):
          st.write(""" - VIP: nhóm khách Vip """)
          st.write(""" - BIG SPENDERS: nhóm khách đóng góp số tiền nhiều nhất """)
          st.write(""" - LOYAL: nhóm khách hàng thân thiết """)
          st.write(""" - POTENTIAL: nhóm khách hàng mới tiềm năng """)
          st.write(""" - ACTIVE: nhóm khách mua không nhiều nhưng hoạt động gần đây nhất """)
          st.write(""" - LOST: nhóm khách hàng chi số tiền ít nhất """)
          st.write(""" - LIGHT: khách hàng đã ngừng mua hàng """)
          st.write(""" - REGULARS: khách hàng thông thường """)
     
     rfm_df = pd.DataFrame({'RFM_level': ['VIP','BIG SPENDERS', 'LOYAL', 'POTENTIAL', 'ACTIVE', 'LOST','LIGHT', 'REGULARS'],
                             'Count': [474, 448, 193, 383, 258, 1079, 452, 1030] })
     st.code(rfm_df)
     st.write(""" Trực quan dữ liệu sau khi áp dụng mô hình RFM: """)
     st.image('image/RFM_model_p3.png')
     st.image('image/RFM_model_2_p3.png')
     st.image('image/RFM_model_3_p3.png')
     st.write(""" Nhận xét: """)
     st.write(""" - VIP, BIG SPENDER, LOYAL và POTENTIAL chiếm 1498 khách hàng và chiếm xấp xỉ **81%** tổng doanh thu của công ty""")
     st.write(""" - Các nhóm còn lại có 2819 khách hàng và đóng góp gần **19%** tổng doanh thu""")

elif choice == 'Xây dựng mô hình: GMM':
     st.subheader('Xây dựng mô hình: GMM')
     st.image('image/gmm_p4_1.jpg')
     st.code(f'Dữ liệu có: {data.shape[0]} dòng và {data.shape[1]} cột')
     st.write(""" Tìm khuỷu phù hợp để xây dựng mô hình: """)
     st.image('image/elbow_p4.png')
     st.write(""" Sau khi áp dụng mô hình, dữ liệu được chia: """)
     st.image('image/gmm_p4.png')
     st.dataframe({'Cluster': ['Cluster0','Cluster1','Cluster2','Cluster3'], 'Mô tả': ['Khách thông thường', 'Khách thân thiết','Khách vip', 'Khách đã bị mất']})
     st.image('image/gmm_visualize_p4.png')
     st.write(""" Doanh thu khách hàng ở 4 nhóm: """)
     st.image('image/sum_money_p4.png')
     st.write(""" Nhận xét:""")
     #st.dataframe({'Cluster': ['Cluster0','Cluster1','Cluster2','Cluster3'], 'Mô tả': ['Khách thông thường', 'Khách thân thiết','Khách vip', 'Khách đã bị mất']})
     st.write(""" - cluster0, cluster1, cluster2 có **1739** khách hàng và chiếm gần **87%** tổng doanh thu của công ty => nên tập trung chăm sóc các nhóm khách hàng này """)
     st.write(""" - cluster3 có **2578** khách hàng, chiếm phần lớn số khách hàng nhưng chỉ đóng góp xấp xỉ **13%** tổng doanh thu""")


elif choice == 'Dự đoán':
     st.subheader('Dự đoán dữ liệu mới')
     flag = False 
     lines = None
     type = st.radio('Chọn: ', options =('Upload file', 'Input data'))

     if type == 'Upload file':
          uploaded_file = st.file_uploader('Chọn file:', type=['txt','csv'])
          if uploaded_file is not None:
               lines = pd.read_csv(uploaded_file, header=0)
               lines = pd.DataFrame(lines)

               scaler = RobustScaler()
               df_scale = scaler.fit_transform(lines)
               df_scale = pd.DataFrame(df_scale, columns=lines.columns)
               with st.expander("Thông tin dữ liệu:"):
                    st.code(lines)
                    st.code(f'Dữ liệu có {df_scale.shape[0]} dòng và {df_scale.shape[1]} cột')

               y_pred_new = gmm_model.predict(df_scale)
               y_pred_new = pd.DataFrame(y_pred_new, columns=['Cluster'])
               result = pd.concat([lines, y_pred_new], axis=1)

               if st.button('Kết quả dự đoán:'):
                    def func(x):
                         if x==0:
                              return 'Khách thông thường'
                         elif x==1:
                              return 'Khách thân thiết'
                         elif x==2: 
                              return 'Khách vip'
                         else:
                              return 'Khách đã bị mất'

                    result['Label'] = result['Cluster'].apply(func)
                    st.code(result)


     if type == 'Input data':
          col1, col2, col3 = st.columns(3)
          with col1:
               recency = st.text_area(label='Nhập Recency:')
          with col2:
               frequency = st.text_area(label='Nhập Frequency:')
          with col3: 
               monetary = st.text_area(label='Nhập Monetary:')
          if st.button('Kết quả dự đoán:'):
               if recency and frequency and monetary:
                    df = pd.DataFrame([{'Recency': recency, 'Frequency': frequency, 'Monetary': monetary}])
                    scaler = RobustScaler()
                    df_scale = scaler.fit_transform(df)
                    df_scale = pd.DataFrame(df_scale, columns=df.columns)          
                    y_pred_new = gmm_model.predict(df)
                    print(y_pred_new)
                    if y_pred_new == 0:
                         st.code('Khách thông thường')
                    elif y_pred_new == 1:
                         st.code('Khách thân thiết')
                    elif y_pred_new == 2:
                         st.code('Khách vip')
                    else:
                         st.code('Khách đã bị mất')
               if not (recency and frequency and monetary):
                    st.code('Nhập lại các giá trị')
