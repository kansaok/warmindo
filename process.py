from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, substring, hour, date_format, month, desc, expr
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Fungsi untuk menampilkan grafik
def show_plot():
    plt.tight_layout()
    plt.show()

# Fungsi untuk mengonversi DataFrame Spark ke DataFrame Pandas
def to_pandas_df(spark_df):
    return spark_df.toPandas()

## A. Pengumpulan Data 
# Inisialisasi Spark Session
spark = SparkSession.builder.appName("Warmindo Sales Analysis").getOrCreate()
# Menetapkan urutan nama hari
days_order = expr("""
    CASE 
        WHEN hari = 'Sunday' THEN 1
        WHEN hari = 'Monday' THEN 2
        WHEN hari = 'Tuesday' THEN 3
        WHEN hari = 'Wednesday' THEN 4
        WHEN hari = 'Thursday' THEN 5
        WHEN hari = 'Friday' THEN 6
        WHEN hari = 'Saturday' THEN 7
    END
""")
# Membaca file CSV
data = spark.read.csv("warmindo.csv", header=True, inferSchema=True)
# Menampilkan schema data
print("SKEMA")
data.printSchema()
# Menampilkan beberapa baris data
print("SAMPLE DATA")
data.show()
# Melihat distribusi jenis produk
print("JENIS PRODUK")
data.groupBy("jenis_produk").count().show()

## B. Pembersihan Data
# Menghapus duplikat
data = data.dropDuplicates()
# Menghapus baris dengan nilai yang hilang
data = data.na.drop()
# Menghapus duplikat
data = data.dropDuplicates()
# Mengidentifikasi dan mengatasi outlier (misalnya, harga jual yang tidak realistis)
data = data.filter((col("harga_jual") > 0) & (col("quantity") > 0))

## C. Transformasi Data
# Mengonversi kolom tanggal_transaksi dari format m/d/yy menjadi yyyy-MM-dd
data = data.withColumn("tanggal_transaksi", to_date(col("tanggal_transaksi"), "M/d/yy"))
# Mengubah kolom tanggal_transaksi menjadi tipe tanggal
data = data.withColumn("tanggal_transaksi", to_date(col("tanggal_transaksi"), "yyyy-MM-dd"))
# Menambahkan kolom bulan_transaksi untuk analisis musiman
data = data.withColumn("bulan_transaksi", substring(col("tanggal_transaksi"), 1, 7))
# Menambah kolom waktu (jam, hari, bulan)
# data = data.withColumn("jam", hour(col("tanggal_transaksi")))
data = data.withColumn("hari", date_format("tanggal_transaksi", "EEEE"))
data = data.withColumn("bulan", month(col("tanggal_transaksi")))

## D. Analisis Data
# 1. Menghitung jenis produk paling laris berdasarkan total penjualan
most_popular_product_type = data.groupBy("jenis_produk").sum("total_penjualan").orderBy(desc("sum(total_penjualan)")).first()
print("JENIS PRODUK PALING LARIS:", most_popular_product_type["jenis_produk"])

print("\n")
print("TOP 5 PRODUK TERLARIS")
# 2. Menghitung top 5 produk terlaris berdasarkan quantity
top_5_best_selling_products = data.groupBy("nama_produk").sum("quantity").orderBy(desc("sum(quantity)")).limit(5)
top_5_best_selling_products.show()

print("TOP 5 PRODUK KURANG LAKU")
# 3. Menghitung top 5 produk kurang laku berdasarkan quantity
top_5_least_selling_products = data.groupBy("nama_produk").sum("quantity").orderBy("sum(quantity)").limit(5)
top_5_least_selling_products.show()

# 4. Tren Penjualan berdasarkan waktu
# # Tren penjualan berdasarkan jam
# sales_by_hour = data.groupBy("jam").sum("total_penjualan").orderBy("jam")
# sales_by_hour.show()
print("\n")
print("TREN PENJUALAN BERDASARKAN WAKTU")
# Tren penjualan berdasarkan hari
sales_by_day = data.groupBy("hari").sum("total_penjualan").orderBy(days_order)
sales_by_day.show()
# Tren penjualan berdasarkan bulan
sales_by_month = data.groupBy("bulan").sum("total_penjualan").orderBy("bulan")
sales_by_month.show()

print("PREFENSI PEMBAYARAN")
# 5. Menghitung preferensi pembayaran berdasarkan jumlah transaksi
payment_preferences = data.groupBy("jenis_pembayaran").count().orderBy(desc("count"))
payment_preferences.show()

print("PERFORMA PENJUALAN BERDASARKAN METODE PEMESANAN")
# 7. Menghitung performa penjualan berdasarkan metode pemesanan
order_method_performance = data.groupBy("jenis_pesanan").sum("total_penjualan").orderBy(desc("sum(total_penjualan)"))
order_method_performance.show()

## E. Pembuatan Model
assembler = VectorAssembler(inputCols=["quantity", "harga_jual"], outputCol="features")
data_vector = assembler.transform(data)
# Membuat model regresi linear untuk memprediksi total penjualan
lr = LinearRegression(featuresCol="features", labelCol="total_penjualan")
lr_model = lr.fit(data_vector)
# Memprediksi total penjualan
predictions = lr_model.transform(data_vector)
predictions.select("total_penjualan", "prediction").show()

## F. Visualisasi Data
# 1. JENIS PRODUK PALING LARIS
plt.figure(figsize=(8, 6))
colors = plt.cm.tab10(np.arange(len(most_popular_product_type)))
bars = plt.bar(most_popular_product_type["jenis_produk"], most_popular_product_type["sum(total_penjualan)"], color=colors)
plt.title("Jenis Produk Paling Laris")
plt.xlabel("Jenis Produk")
plt.ylabel("Total Penjualan")
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, 
             f'{int(yval):,}', ha='center', va='bottom', fontsize=10)
show_plot()

# 2. TOP 5 PRODUK TERLARIS
top_5_pd = to_pandas_df(top_5_best_selling_products)
colors = plt.cm.tab10(np.arange(len(top_5_pd)))
plt.figure(figsize=(10, 6))
bars = plt.bar(top_5_pd["nama_produk"], top_5_pd["sum(quantity)"], color=colors)
plt.title("Top 5 Produk Terlaris")
plt.xlabel("Nama Produk")
plt.ylabel("Jumlah Terjual")
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
plt.xticks(rotation=30)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, 
             f'{int(yval):,}', ha='center', va='bottom', fontsize=10)
show_plot()

# 3. TOP 5 PRODUK KURANG LAKU
top_5_least_pd = to_pandas_df(top_5_least_selling_products)
colors = plt.cm.tab10(np.arange(len(top_5_least_pd)))
plt.figure(figsize=(10, 6))
bars = plt.bar(top_5_least_pd["nama_produk"], top_5_least_pd["sum(quantity)"], color=colors)
plt.title("Top 5 Produk Kurang Laku")
plt.xlabel("Nama Produk")
plt.ylabel("Jumlah Terjual")
plt.xticks(rotation=30)
for bar in bars:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
             int(bar.get_height()), ha='center', va='bottom')
show_plot()

# 4. TREN PENJUALAN BERDASARKAN WAKTU
# HARI
sales_by_day_pd = to_pandas_df(sales_by_day)
colors = plt.cm.tab10(np.arange(len(sales_by_day_pd)))
plt.figure(figsize=(10, 6))
bars = plt.bar(sales_by_day_pd["hari"], sales_by_day_pd["sum(total_penjualan)"], color=colors)
plt.title("Tren Penjualan Berdasarkan Hari")
plt.xlabel("Hari")
plt.ylabel("Total Penjualan")
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, 
             f'{int(yval):,}', ha='center', va='bottom', fontsize=10)
show_plot()

# BULAN
sales_by_month_pd = to_pandas_df(sales_by_month)
plt.figure(figsize=(10, 6))
plt.plot(sales_by_month_pd["bulan"], sales_by_month_pd["sum(total_penjualan)"], marker='o')
plt.title("Tren Penjualan Berdasarkan Bulan")
plt.xlabel("Bulan")
plt.ylabel("Total Penjualan")
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
for i, val in enumerate(sales_by_month_pd["sum(total_penjualan)"]):
    plt.text(sales_by_month_pd["bulan"][i], val, f'{int(val):,}', ha='center', va='bottom')
show_plot()

# 5. PREFENSI PEMBAYARAN
payment_preferences_pd = to_pandas_df(payment_preferences)
colors = plt.cm.tab10(np.arange(len(payment_preferences_pd)))
plt.figure(figsize=(10, 6))
bars = plt.bar(payment_preferences_pd["jenis_pembayaran"], payment_preferences_pd["count"], color=colors)
plt.title("Preferensi Pembayaran")
plt.xlabel("Jenis Pembayaran")
plt.ylabel("Jumlah Transaksi")
for bar in bars:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
             int(bar.get_height()), ha='center', va='bottom')
show_plot()

# 6. PERFORMA PENJUALAN BERDASARKAN METODE PEMESANAN
order_method_performance_pd = to_pandas_df(order_method_performance)
colors = plt.cm.tab10(np.arange(len(order_method_performance_pd)))
plt.figure(figsize=(10, 6))
bars = plt.bar(order_method_performance_pd["jenis_pesanan"], order_method_performance_pd["sum(total_penjualan)"], color=colors)
plt.title("Performa Penjualan Berdasarkan Metode Pemesanan")
plt.xlabel("Jenis Pesanan")
plt.ylabel("Total Penjualan")
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, 
             f'{int(yval):,}', ha='center', va='bottom', fontsize=10)
show_plot()

# Stop the SparkSession
spark.stop()