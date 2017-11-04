package com.example.machinceLearning;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.recommendation.ALS;
import org.apache.spark.ml.recommendation.ALSModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import com.example.common.utility.SparkConnection;

public class SparkMLRecommendationsDemo {



	public static void main(String[] args) {
		
		Logger.getLogger("org").setLevel(Level.ERROR);
		Logger.getLogger("akka").setLevel(Level.ERROR);
		JavaSparkContext spContext = SparkConnection.getContext();
		SparkSession spSession = SparkConnection.getSession();
		
		/*--------------------------------------------------------------------------
		Load Data
		--------------------------------------------------------------------------*/
		
		Dataset<Row> rawDf = spSession.read()
							.csv("data/UserItemData.txt");
		rawDf.show(5);
		rawDf.printSchema();
		
		StructType ratingsSchema = DataTypes
				.createStructType(new StructField[] {
						DataTypes.createStructField("user", DataTypes.IntegerType, false),
						DataTypes.createStructField("item", DataTypes.IntegerType, false),
						DataTypes.createStructField("rating", DataTypes.DoubleType, false) 
					});
		
		JavaRDD<Row> rdd1 = rawDf.toJavaRDD().repartition(2);
		

		JavaRDD<Row> rdd2 = rdd1.map( new Function<Row, Row>() {

			@Override
			public Row call(Row iRow) throws Exception {
				
				Row retRow = RowFactory.create( 
								Integer.valueOf(iRow.getString(0)),
								Integer.valueOf(iRow.getString(1)), 
								Double.valueOf(iRow.getString(2)) );
				
				return retRow;
			}

		});

		Dataset<Row> ratingsDf = spSession.createDataFrame(rdd2, ratingsSchema);
		System.out.println("Ratings Data: ");
		ratingsDf.show(5);

		Dataset<Row>[] splits = ratingsDf.randomSplit(new double[]{0.9, 0.1});
		Dataset<Row> training = splits[0];
		Dataset<Row> test = splits[1];
		
		ALS als = new ALS()
				  .setMaxIter(5)
				  .setRegParam(0.01)
				  .setUserCol("user")
				  .setItemCol("item")
				  .setRatingCol("rating");
		
		ALSModel model = als.fit(training);


		Dataset<Row> predictions = model.transform(test);
		
		System.out.println("Predictions : ");
		predictions.show();
		

		com.example.common.utility.ExerciseUtils.hold();
	}


}
