from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, count, avg
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Création d'une session Spark
spark = SparkSession.builder \
    .appName("Analyse de Données de Ventes") \
    .getOrCreate()

# Chargement des données de ventes à partir d'un fichier CSV
sales_data = spark.read.csv('sales_data.csv', header=True, inferSchema=True)

# Affichage des premières lignes des données et des statistiques descriptives
print("Premières lignes des données :")
sales_data.show(5)

print("Statistiques descriptives :")
sales_data.describe().show()

# Calcul du chiffre d'affaires total
total_revenue = sales_data.select(sum(col('revenue'))).collect()[0][0]
print("Chiffre d'affaires total : $" + str(total_revenue))

# Calcul du nombre total de commandes
total_orders = sales_data.select(count(col('order_id'))).collect()[0][0]
print("Nombre total de commandes : " + str(total_orders))

# Calcul de la valeur moyenne des commandes
average_order_value = sales_data.select(avg(col('revenue'))).collect()[0][0]
print("Valeur moyenne des commandes : $" + str(average_order_value))

# Analyse des ventes par région
sales_by_region = sales_data.groupBy('region') \
    .agg(sum('revenue').alias('total_revenue'), count('order_id').alias('total_orders')) \
    .orderBy('total_revenue', ascending=False)
print("Analyse des ventes par région :")
sales_by_region.show()

# Visualisation des résultats avec des graphiques
# (Code pour créer des graphiques avec Matplotlib, Plotly, etc.)

# Fonction pour prétraiter les données de ventes
def preprocess_sales_data(data):
    # Nettoyage des données (ex. suppression des valeurs manquantes)
    cleaned_data = data.dropna()
    # Transformation des types de données si nécessaire
    transformed_data = cleaned_data.withColumn('date', col('date').cast('date'))
    return transformed_data

# Fonction pour analyser les ventes par catégorie de produit
def analyze_sales_by_category(data):
    # Analyse des ventes par catégorie de produit
    sales_by_category = data.groupBy('category') \
        .agg(sum('revenue').alias('total_revenue')) \
        .orderBy('total_revenue', ascending=False)
    return sales_by_category

# Fonction pour entraîner un modèle de régression linéaire de prédiction de revenu
def train_linear_regression_model(train_data):
    # Préparation des données pour l'entraînement du modèle
    assembler = VectorAssembler(inputCols=['quantity'], outputCol='features')
    train_data = assembler.transform(train_data)
    
    # Création du modèle de régression linéaire
    lr = LinearRegression(featuresCol='features', labelCol='revenue')
    
    # Entraînement du modèle
    lr_model = lr.fit(train_data)
    
    return lr_model

# Fonction pour évaluer les performances du modèle sur les données de test
def evaluate_model_performance(model, test_data):
    # Préparation des données pour l'évaluation du modèle
    test_data = assembler.transform(test_data)
    
    # Prédiction sur les données de test
    predictions = model.transform(test_data)
    
    # Calcul des métriques de performance (RMSE et R2)
    evaluator = RegressionEvaluator(labelCol='revenue', predictionCol='prediction', metricName='rmse')
    rmse = evaluator.evaluate(predictions)
    
    evaluator = RegressionEvaluator(labelCol='revenue', predictionCol='prediction', metricName='r2')
    r2 = evaluator.evaluate(predictions)
    
    return rmse, r2

# Division des données en ensembles de formation et de test
train_data, test_data = cleaned_sales_data.randomSplit([0.8, 0.2], seed=123)

# Entraînement du modèle de régression linéaire
linear_regression_model = train_linear_regression_model(train_data)

# Évaluation des performances du modèle sur les données de test
rmse, r2 = evaluate_model_performance(linear_regression_model, test_data)

# Affichage des métriques de performance
print("Métriques de performance du modèle de régression linéaire :")
print("RMSE :", rmse)
print("R2   :", r2)



# Appel des fonctions pour prétraiter et analyser les données de ventes
cleaned_sales_data = preprocess_sales_data(sales_data)
sales_by_category = analyze_sales_by_category(cleaned_sales_data)

# Affichage des résultats de l'analyse par catégorie de produit
print("Analyse des ventes par catégorie de produit :")
sales_by_category.show()

# Test des performances du modèle sur les données de ventes
test_model_performance(cleaned_sales_data)

# Arrêt de la session Spark
spark.stop()
