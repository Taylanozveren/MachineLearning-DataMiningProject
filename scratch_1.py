import numpy as np
import pandas as pd
from datetime import datetime

# CSV dosyasını yükleyin
df = pd.read_excel("C:\\Users\\Win10\\Desktop\\ACM476 Data Mining Course\\OnlineRetail.xlsx")
print(df.head())  # İlk 5 satırı görüntüle

# Veri seti hakkında genel bilgi
print("Data Shape:", df.shape)
print(df.info())  #data info
print("Column Names:", df.columns)

# Eksik verileri kontrol etme
print(df.isnull().sum())

# Betimleyici istatistikler
print(df.describe(include='all'))

# Eksik CustomerID satırlarını kaldır
df = df.dropna(subset=['CustomerID'])
print(f"After dropping rows with missing CustomerID, data shape: {df.shape}")

# Aynı StockCode'a sahip satırlardaki Description değerini kullanarak eksik değerleri doldurma
df['Description'] = df.groupby('StockCode')['Description'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else ''))

# Yinelenen satırları silmeden önce kontrol etme
# Yinelenen satırları belirleyin ve ayrı bir veri çerçevesine kaydedin
duplicates = df[df.duplicated()]

# Yinelenen satırların sayısını ve bazı örneklerini inceleyin
print(f"Number of duplicate rows: {duplicates.shape[0]}")
print(duplicates.head())

# İsteğe bağlı olarak yinelenen satırları bir dosyaya kaydedin
duplicates.to_csv("duplicates.csv", index=False)

# Ardından yinelenen satırları silin
df = df.drop_duplicates()
print(f"After removing duplicate rows, data shape: {df.shape}")

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Sayısal sütunları seçme
numeric_df = df.select_dtypes(include=[np.number])

# Korelasyon matrisini oluşturma ve mutlak değer alma
cor_matrix = numeric_df.corr().abs()

# Korelasyon matrisini görselleştirme
plt.figure(figsize=(12, 12))
sns.heatmap(cor_matrix, cmap='RdBu', annot=True)
plt.show()

    # Üst üçgen matrisinde varsa 0.90’dan büyük korelasyona sahip sütunları belirleyip kaldırıyoruz
    # Bu aşamayı yapmamızın ana nedeni, yüksek korelasyon gösteren sütunları kaldırarak veri setindeki multicollinearity (çoklu doğrusal bağlantı) sorununu azaltmaktır.
    # Modelde fazla bilgi taşıyan, benzer bilgiyi tekrar eden değişkenleri çıkarmış oluyoruz. Bu da modelin daha sağlıklı, hızlı ve genellenebilir hale gelmesini sağlar.

    # Üst üçgende yüksek korelasyon (örneğin, >0.90) olan sütunları kontrol etme
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    high_corr = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > 0.90)]

    # Sonuçları gösterme
    print("Yüksek korelasyonlu sütunlar:", high_corr)

# Aykırı değer temizleme fonksiyonu
def remove_outliers(df, column, multiplier=3.0):  # Varsayılan multiplier = 3.0 (daha az veri kaybı için)
    Q1 = df[column].quantile(0.25)  # 1. Çeyrek
    Q3 = df[column].quantile(0.75)  # 3. Çeyrek
    IQR = Q3 - Q1  # IQR Hesaplama
    lower_bound = Q1 - multiplier * IQR  # Alt sınır
    upper_bound = Q3 + multiplier * IQR  # Üst sınır
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Tüm sayısal sütunları seç
numeric_columns = df.select_dtypes(include=[np.number]).columns

# Aykırı değerleri temizleme işlemi
for col in numeric_columns:
    df = remove_outliers(df, col, multiplier=3.0)  # 3.0 IQR ile veri kaybını azalt

# Aykırı değerlerden sonra her sayısal sütun için boxplot çizdirme
for col in numeric_columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df[col])
    plt.title(f"{col} için Boxplot (Aykırı Değerler Temizlenmiş)")
    plt.show()

# Son durumun boyutunu ve betimleyici istatistiklerini yazdırma
print("Aykırı değer temizlendikten sonra boyut:", df.shape)
print(df.describe())


    # Bağımlı değişkene göre özet istatistik çıkarma fonksiyonu
    def target_summary_with_num(dataframe, target, num_col):
        """
        Shows the mean of 'num_col' numerical variable based on a specified 'target' categorical variable.
        """
        summary = dataframe.groupby(target).agg({num_col: 'mean'})
        print(f"\nAverage {num_col} by {target}:\n", summary, "\n")
        return summary


    def calculate_total_price(dataframe):
        """
        Creates a total price column, calculates the count of transactions above and below the average,
        and performs right-skewness analysis.
        """
        # TotalPrice hesaplama
        dataframe['TotalPrice'] = dataframe['Quantity'] * dataframe['UnitPrice']

        # Ortalama üstü ve altı işlem sayıları
        mean_total_price = dataframe['TotalPrice'].mean()
        above_mean_count = dataframe[dataframe['TotalPrice'] > mean_total_price]['TotalPrice'].count()
        below_mean_count = dataframe[dataframe['TotalPrice'] < mean_total_price]['TotalPrice'].count()

        print("Number of transactions above average:", above_mean_count)
        print("Number of transactions below average:", below_mean_count)

        return dataframe

    # TotalPrice hesaplama ve analiz fonksiyonunu çalıştırma
    df = calculate_total_price(df)

    # Ülke bazında ortalama sipariş miktarı (Quantity)
    target_summary_with_num(df, 'Country', 'Quantity')

    # Ülke bazında ortalama toplam harcama (TotalPrice)
    target_summary_with_num(df, 'Country', 'TotalPrice')

    # Müşteri bazında ortalama harcama miktarı
    target_summary_with_num(df, 'CustomerID', 'TotalPrice')

    # Fatura bazında toplam harcama miktarı
    target_summary_with_num(df, 'InvoiceNo', 'TotalPrice')

    # Ürün bazında ortalama sipariş miktarı
    target_summary_with_num(df, 'StockCode', 'Quantity')

    # Yıl bazında toplam harcama miktarı
    df['Year'] = df['InvoiceDate'].dt.year  # Yıl sütununu oluşturma
    target_summary_with_num(df, 'Year', 'TotalPrice')

    # 1. TotalPrice sütununun mevcut olup olmadığını kontrol et ve hesapla
    # TotalPrice = Quantity * UnitPrice
    if 'TotalPrice' not in df.columns:
        df['TotalPrice'] = df['Quantity'] * df['UnitPrice']  # Satış işlemleri için toplam tutar hesaplama
        print(df[['Quantity', 'UnitPrice', 'TotalPrice']].head())

    # 2. Negatif TotalPrice ve Quantity değerlerinin sayısını hesapla
    negative_total_price_count = df[df['TotalPrice'] < 0].shape[0]  # Negatif TotalPrice satır sayısı
    negative_quantity_count = df[df['Quantity'] < 0].shape[0]  # Negatif Quantity satır sayısı
    print("Number of negative TotalPrice values:", negative_total_price_count)
    print("Number of negative Quantity values:", negative_quantity_count)

    # 3. İade işlemlerini ayır ve ayrı bir DataFrame oluştur
    # İade işlemleri: Negatif Quantity veya TotalPrice değerleri
    returns_df = df[(df['Quantity'] < 0) | (df['TotalPrice'] < 0)]  # İade işlemleri
    non_returns_df = df[(df['Quantity'] >= 0) & (df['TotalPrice'] >= 0)]  # Satış işlemleri

    # 4. Müşteri bazında toplam satın alma ve iade miktarlarını hesapla
    # Toplam satın alma miktarı
        total_quantity = non_returns_df.groupby('CustomerID')['Quantity'].sum()  # Her müşteri için toplam satın alınan miktar

    # Toplam iade edilen miktar
    returned_quantity = returns_df.groupby('CustomerID')['Quantity'].sum().abs()  # İade edilen miktarın mutlak değeri

    # 5. İade oranını hesapla
    # İade oranı: iade edilen miktarın toplam satın alınan miktara oranı
    return_rate = (returned_quantity / total_quantity).fillna(0)  # Toplam miktara bölerek oran hesapla

    # 6. Müşteri başına iade sayısı ve toplam iade tutarını hesapla
    # İade edilen toplam fatura sayısı
    return_count = returns_df.groupby('CustomerID').size()  # Her müşteri için toplam iade edilen işlem sayısı

    # Toplam iade tutarı (TotalPrice üzerinden)
    total_return_amount = returns_df.groupby('CustomerID')['TotalPrice'].sum().abs()  # İade edilen toplam tutar

    # 7. Özellikleri orijinal veri setine ekle
    # İade ile ilgili hesaplanan değerleri orijinal df'e ekliyoruz
    df['ReturnRate'] = df['CustomerID'].map(return_rate)  # İade oranını orijinal df'e ekle
    df['ReturnCount'] = df['CustomerID'].map(return_count).fillna(0)  # İade sayısını ekle (NaN yerine 0)
    df['TotalReturnAmount'] = df['CustomerID'].map(total_return_amount).fillna(0)  # Toplam iade tutarını ekle

    # 8. Pozitif değerler için log dönüşümü
    # Log dönüşümü, çarpıklığı (skewness) azaltır ve veriyi daha dengeli hale getirir
    df['LogTotalPrice'] = np.log1p(df['TotalPrice'].where(df['TotalPrice'] > 0, 0))  # TotalPrice log dönüşümü
    df['LogQuantity'] = np.log1p(df['Quantity'].where(df['Quantity'] > 0, 0))  # Quantity log dönüşümü

    # 9. Log dönüşümlü sütunların çarpıklık değerlerini hesapla ve normal dağılım olarak simetrik hale getirdim
    # Çarpıklık (skewness) ne kadar düşükse, dağılım o kadar simetriktir
    log_total_price_skewness = df['LogTotalPrice'].skew()  # LogTotalPrice çarpıklık değeri
    log_quantity_skewness = df['LogQuantity'].skew()  # LogQuantity çarpıklık değeri
    print("LogTotalPrice Skewness Value:", log_total_price_skewness)
    print("LogQuantity Skewness Value:", log_quantity_skewness)

    # 10. Görselleştirme: LogTotalPrice ve ReturnRate dağılımı

    # LogTotalPrice dağılımını çiz
    plt.figure(figsize=(8, 5))
    sns.histplot(df['LogTotalPrice'].dropna(), bins=30, kde=True)
    plt.title('LogTotalPrice Dağılımı')
    plt.xlabel('LogTotalPrice')
    plt.ylabel('Frekans')
    plt.show()

    # ReturnRate dağılımını çiz
    plt.figure(figsize=(8, 5))
    sns.histplot(df['ReturnRate'].dropna(), bins=30, kde=True)
    plt.title('İade Oranı Dağılımı')
    plt.xlabel('ReturnRate')
    plt.ylabel('Frekans')
    plt.show()


high_return_customers = df[df['ReturnRate'] > 1]
print(high_return_customers[['CustomerID', 'ReturnRate', 'ReturnCount', 'TotalReturnAmount']])
## Bu müşterilerde, satın alma kayıtlarının eksikliği veya iade işlemlerinin birden fazla kez kaydedilmiş olması gibi bir veri sorunu olabilir.


# Son EDA işlemlerinden sonra veri setinin analizini yapıyorum
print(df.shape)
df.info()
print(df.columns)