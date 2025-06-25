import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import random
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SwiggyReviewAnalyzer:
    def __init__(self):
        self.reviews_df = None
        self.sentiment_scores = None
        
    def generate_dummy_reviews(self, num_reviews=500):
        """Generate realistic dummy Swiggy reviews"""
        
        positive_reviews = [
            "Amazing food quality! Order arrived hot and fresh. Definitely ordering again.",
            "Super fast delivery and the biryani was absolutely delicious. 5 stars!",
            "Great service by Swiggy. Food was packed well and delivery was on time.",
            "Loved the variety of restaurants available. My pizza was perfect!",
            "Excellent customer service. They resolved my issue quickly and professionally.",
            "Best food delivery app! Always reliable and food quality is top-notch.",
            "Quick delivery even during peak hours. The burger was amazing!",
            "Fresh ingredients, hot food, and friendly delivery partner. Highly recommended!",
            "Swiggy never disappoints! The Chinese food was exactly what I wanted.",
            "Perfect packaging and the dessert reached in great condition. Loved it!",
            "Outstanding service! Food arrived 10 minutes early and was delicious.",
            "Great app interface and easy ordering process. Food was fantastic!",
            "Delivery partner was very polite and food quality exceeded expectations.",
            "Amazing deals and discounts! Got great food at reasonable prices.",
            "Reliable service as always. The South Indian meal was authentic and tasty.",
            "Excellent food temperature maintenance. Everything was fresh and hot!",
            "Swiggy's customer support is remarkable. Quick response and helpful.",
            "Love the live tracking feature. Food arrived exactly when promised.",
            "Great variety of cuisines available. The Italian food was restaurant quality!",
            "Perfect portion sizes and amazing taste. Worth every penny spent!"
        ]
        
        negative_reviews = [
            "Terrible experience! Food arrived cold and the order was completely wrong.",
            "Worst delivery service ever. Waited 2 hours for cold, stale food.",
            "Food quality was horrible. Seemed like it was prepared days ago.",
            "Delivery partner was rude and unprofessional. Very disappointing service.",
            "Order got cancelled after 1 hour of waiting. Completely unreliable!",
            "Food was spilled all over the bag. Poor packaging and careless handling.",
            "Overpriced food with terrible taste. Will never order from here again.",
            "Customer service is non-existent. No response to my complaints.",
            "Food arrived 3 hours late and was completely inedible. Waste of money!",
            "Wrong order delivered and when I called, they were not helpful at all.",
            "Poor hygiene standards. Found hair in my food. Absolutely disgusting!",
            "App keeps crashing during payment. Technical issues are frustrating.",
            "Delivery partner couldn't find my address despite clear instructions.",
            "Food was burnt and tasted awful. How can they serve such quality?",
            "Charged extra fees without any prior notice. Very misleading pricing.",
            "Restaurant cancelled order after 1.5 hours. Completely wasted my time.",
            "Food packaging was torn and contents were spilled. Very unprofessional.",
            "Delivery took forever and food was stone cold when it arrived.",
            "Poor quality ingredients used. The meal was tasteless and overpriced.",
            "Worst customer experience. Will switch to other food delivery apps."
        ]
        
        neutral_reviews = [
            "Food was okay, nothing special. Delivery was on time though.",
            "Average experience. Food taste was decent but could be better.",
            "Delivery was fine but food quality was just average for the price.",
            "Okay service overall. Some items were good, others were mediocre.",
            "Food arrived on time but wasn't as hot as expected. Taste was okay.",
            "Decent app interface but food quality varies by restaurant.",
            "Service is usually good but this time it was just average.",
            "Food was acceptable but portion sizes could be larger for the price.",
            "Mixed experience - some dishes were good, others were disappointing.",
            "Delivery partner was polite but food quality was just okay.",
            "App works fine but food took longer than estimated time.",
            "Food quality is inconsistent - sometimes great, sometimes average.",
            "Reasonable prices but food taste is hit or miss depending on restaurant.",
            "Packaging was good but food temperature could have been better.",
            "Customer service responded but resolution took longer than expected."
        ]
        
        restaurants = [
            "McDonald's", "KFC", "Pizza Hut", "Domino's", "Subway", "Burger King",
            "Taco Bell", "Biryani House", "Chinese Garden", "South Indian Corner",
            "Italian Bistro", "Cafe Coffee Day", "Starbucks", "Haldiram's",
            "Barbeque Nation", "Faasos", "Behrouz Biryani", "Oven Story Pizza",
            "Wow! Momo", "The Belgian Waffle Co."
        ]
        
        cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Pune", "Kolkata"]
        
        reviews_data = []
        
        # Generate positive reviews (50%)
        for i in range(int(num_reviews * 0.5)):
            review = {
                'review_id': f'R{i+1:04d}',
                'review_text': random.choice(positive_reviews),
                'rating': random.choice([4, 5]),
                'restaurant': random.choice(restaurants),
                'city': random.choice(cities),
                'delivery_time': random.randint(20, 45),
                'order_value': random.randint(200, 800)
            }
            reviews_data.append(review)
        
        # Generate negative reviews (30%)
        start_idx = int(num_reviews * 0.5)
        for i in range(int(num_reviews * 0.3)):
            review = {
                'review_id': f'R{start_idx + i + 1:04d}',
                'review_text': random.choice(negative_reviews),
                'rating': random.choice([1, 2]),
                'restaurant': random.choice(restaurants),
                'city': random.choice(cities),
                'delivery_time': random.randint(60, 180),
                'order_value': random.randint(150, 600)
            }
            reviews_data.append(review)
        
        # Generate neutral reviews (20%)
        start_idx = int(num_reviews * 0.8)
        for i in range(int(num_reviews * 0.2)):
            review = {
                'review_id': f'R{start_idx + i + 1:04d}',
                'review_text': random.choice(neutral_reviews),
                'rating': 3,
                'restaurant': random.choice(restaurants),
                'city': random.choice(cities),
                'delivery_time': random.randint(30, 60),
                'order_value': random.randint(180, 500)
            }
            reviews_data.append(review)
        
        # Shuffle the reviews
        random.shuffle(reviews_data)
        
        # Create DataFrame
        self.reviews_df = pd.DataFrame(reviews_data)
        print(f"✅ Generated {len(self.reviews_df)} dummy Swiggy reviews")
        return self.reviews_df
    
    def perform_sentiment_analysis(self):
        """Perform sentiment analysis using TextBlob"""
        if self.reviews_df is None:
            raise ValueError("No reviews data found. Please generate reviews first.")
        
        print("🔍 Performing sentiment analysis...")
        
        # Calculate sentiment scores
        sentiments = []
        polarities = []
        subjectivities = []
        
        for text in self.reviews_df['review_text']:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Classify sentiment based on polarity
            if polarity > 0.1:
                sentiment = 'Positive'
            elif polarity < -0.1:
                sentiment = 'Negative'
            else:
                sentiment = 'Neutral'
            
            sentiments.append(sentiment)
            polarities.append(polarity)
            subjectivities.append(subjectivity)
        
        # Add sentiment data to DataFrame
        self.reviews_df['sentiment'] = sentiments
        self.reviews_df['polarity'] = polarities
        self.reviews_df['subjectivity'] = subjectivities
        
        print("✅ Sentiment analysis completed!")
        return self.reviews_df
    
    def generate_summary_statistics(self):
        """Generate comprehensive summary statistics"""
        if self.reviews_df is None:
            raise ValueError("No reviews data found.")
        
        print("\n" + "="*60)
        print("📊 SWIGGY REVIEWS SENTIMENT ANALYSIS SUMMARY")
        print("="*60)
        
        # Basic statistics
        total_reviews = len(self.reviews_df)
        avg_rating = self.reviews_df['rating'].mean()
        avg_delivery_time = self.reviews_df['delivery_time'].mean()
        avg_order_value = self.reviews_df['order_value'].mean()
        
        print(f"📈 OVERVIEW:")
        print(f"   Total Reviews: {total_reviews}")
        print(f"   Average Rating: {avg_rating:.2f}/5")
        print(f"   Average Delivery Time: {avg_delivery_time:.1f} minutes")
        print(f"   Average Order Value: ₹{avg_order_value:.0f}")
        
        # Sentiment distribution
        sentiment_counts = self.reviews_df['sentiment'].value_counts()
        print(f"\n🎯 SENTIMENT DISTRIBUTION:")
        for sentiment, count in sentiment_counts.items():
            percentage = (count / total_reviews) * 100
            print(f"   {sentiment}: {count} ({percentage:.1f}%)")
        
        # Rating distribution
        print(f"\n⭐ RATING DISTRIBUTION:")
        rating_counts = self.reviews_df['rating'].value_counts().sort_index()
        for rating, count in rating_counts.items():
            percentage = (count / total_reviews) * 100
            stars = "★" * rating + "☆" * (5 - rating)
            print(f"   {rating} {stars}: {count} ({percentage:.1f}%)")
        
        # Top restaurants by review count
        print(f"\n🏪 TOP RESTAURANTS BY REVIEW COUNT:")
        top_restaurants = self.reviews_df['restaurant'].value_counts().head(5)
        for restaurant, count in top_restaurants.items():
            print(f"   {restaurant}: {count} reviews")
        
        # City-wise analysis
        print(f"\n🌆 CITY-WISE SENTIMENT:")
        city_sentiment = pd.crosstab(self.reviews_df['city'], self.reviews_df['sentiment'])
        for city in city_sentiment.index:
            positive_pct = (city_sentiment.loc[city, 'Positive'] / city_sentiment.loc[city].sum()) * 100
            print(f"   {city}: {positive_pct:.1f}% positive")
        
        return {
            'total_reviews': total_reviews,
            'avg_rating': avg_rating,
            'sentiment_distribution': sentiment_counts.to_dict(),
            'rating_distribution': rating_counts.to_dict()
        }
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        if self.reviews_df is None:
            raise ValueError("No reviews data found.")
        
        print("📊 Creating visualizations...")
        
        # Set up the plotting area
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Sentiment Distribution Pie Chart
        plt.subplot(3, 3, 1)
        sentiment_counts = self.reviews_df['sentiment'].value_counts()
        colors = ['#2ecc71', '#e74c3c', '#95a5a6']
        plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
        plt.title('Sentiment Distribution', fontsize=14, fontweight='bold')
        
        # 2. Rating Distribution Bar Chart
        plt.subplot(3, 3, 2)
        rating_counts = self.reviews_df['rating'].value_counts().sort_index()
        bars = plt.bar(rating_counts.index, rating_counts.values, color='#3498db', alpha=0.7)
        plt.xlabel('Rating')
        plt.ylabel('Number of Reviews')
        plt.title('Rating Distribution', fontsize=14, fontweight='bold')
        plt.xticks(range(1, 6))
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{int(height)}', ha='center', va='bottom')
        
        # 3. Sentiment vs Rating Scatter Plot
        plt.subplot(3, 3, 3)
        sentiment_colors = {'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#95a5a6'}
        for sentiment in self.reviews_df['sentiment'].unique():
            data = self.reviews_df[self.reviews_df['sentiment'] == sentiment]
            plt.scatter(data['rating'], data['polarity'], 
                       c=sentiment_colors[sentiment], label=sentiment, alpha=0.6)
        plt.xlabel('Rating')
        plt.ylabel('Polarity Score')
        plt.title('Sentiment vs Rating', fontsize=14, fontweight='bold')
        plt.legend()
        
        # 4. Delivery Time vs Sentiment
        plt.subplot(3, 3, 4)
        sns.boxplot(data=self.reviews_df, x='sentiment', y='delivery_time')
        plt.title('Delivery Time by Sentiment', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        
        # 5. Top Restaurants by Average Rating
        plt.subplot(3, 3, 5)
        restaurant_ratings = self.reviews_df.groupby('restaurant')['rating'].mean().sort_values(ascending=False).head(8)
        bars = plt.barh(range(len(restaurant_ratings)), restaurant_ratings.values, color='#f39c12')
        plt.yticks(range(len(restaurant_ratings)), restaurant_ratings.index)
        plt.xlabel('Average Rating')
        plt.title('Top Restaurants by Average Rating', fontsize=14, fontweight='bold')
        
        # Add value labels
        for i, v in enumerate(restaurant_ratings.values):
            plt.text(v + 0.05, i, f'{v:.2f}', va='center')
        
        # 6. City-wise Sentiment Distribution
        plt.subplot(3, 3, 6)
        city_sentiment = pd.crosstab(self.reviews_df['city'], self.reviews_df['sentiment'])
        city_sentiment.plot(kind='bar', stacked=True, ax=plt.gca(), 
                           color=['#2ecc71', '#e74c3c', '#95a5a6'])
        plt.title('City-wise Sentiment Distribution', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.legend(title='Sentiment')
        
        # 7. Order Value vs Sentiment
        plt.subplot(3, 3, 7)
        sns.violinplot(data=self.reviews_df, x='sentiment', y='order_value')
        plt.title('Order Value Distribution by Sentiment', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        
        # 8. Polarity Distribution
        plt.subplot(3, 3, 8)
        plt.hist(self.reviews_df['polarity'], bins=30, color='#9b59b6', alpha=0.7, edgecolor='black')
        plt.xlabel('Polarity Score')
        plt.ylabel('Frequency')
        plt.title('Polarity Score Distribution', fontsize=14, fontweight='bold')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        
        # 9. Subjectivity vs Polarity
        plt.subplot(3, 3, 9)
        scatter = plt.scatter(self.reviews_df['subjectivity'], self.reviews_df['polarity'], 
                             c=self.reviews_df['rating'], cmap='viridis', alpha=0.6)
        plt.xlabel('Subjectivity')
        plt.ylabel('Polarity')
        plt.title('Subjectivity vs Polarity (Color = Rating)', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, label='Rating')
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Visualizations created successfully!")
    
    def analyze_key_insights(self):
        """Generate key insights and recommendations"""
        if self.reviews_df is None:
            raise ValueError("No reviews data found.")
        
        print("\n" + "="*60)
        print("🔍 KEY INSIGHTS & RECOMMENDATIONS")
        print("="*60)
        
        # Sentiment-based insights
        positive_reviews = self.reviews_df[self.reviews_df['sentiment'] == 'Positive']
        negative_reviews = self.reviews_df[self.reviews_df['sentiment'] == 'Negative']
        
        print("✅ POSITIVE ASPECTS:")
        if len(positive_reviews) > 0:
            avg_positive_delivery = positive_reviews['delivery_time'].mean()
            avg_positive_rating = positive_reviews['rating'].mean()
            print(f"   • {len(positive_reviews)} positive reviews ({len(positive_reviews)/len(self.reviews_df)*100:.1f}%)")
            print(f"   • Average delivery time for positive reviews: {avg_positive_delivery:.1f} minutes")
            print(f"   • Average rating for positive reviews: {avg_positive_rating:.2f}/5")
            print(f"   • Top performing restaurant: {positive_reviews['restaurant'].mode().iloc[0]}")
        
        print("\n❌ AREAS FOR IMPROVEMENT:")
        if len(negative_reviews) > 0:
            avg_negative_delivery = negative_reviews['delivery_time'].mean()
            avg_negative_rating = negative_reviews['rating'].mean()
            print(f"   • {len(negative_reviews)} negative reviews ({len(negative_reviews)/len(self.reviews_df)*100:.1f}%)")
            print(f"   • Average delivery time for negative reviews: {avg_negative_delivery:.1f} minutes")
            print(f"   • Average rating for negative reviews: {avg_negative_rating:.2f}/5")
            print(f"   • Most complained restaurant: {negative_reviews['restaurant'].mode().iloc[0]}")
        
        # Delivery time analysis
        print(f"\n🚚 DELIVERY INSIGHTS:")
        fast_delivery = self.reviews_df[self.reviews_df['delivery_time'] <= 30]
        slow_delivery = self.reviews_df[self.reviews_df['delivery_time'] > 60]
        
        print(f"   • Orders delivered ≤30 min: {len(fast_delivery)} ({len(fast_delivery)/len(self.reviews_df)*100:.1f}%)")
        print(f"   • Orders delivered >60 min: {len(slow_delivery)} ({len(slow_delivery)/len(self.reviews_df)*100:.1f}%)")
        
        if len(fast_delivery) > 0:
            fast_positive_pct = len(fast_delivery[fast_delivery['sentiment'] == 'Positive']) / len(fast_delivery) * 100
            print(f"   • Positive sentiment for fast delivery: {fast_positive_pct:.1f}%")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print("   1. 🎯 Focus on reducing delivery times - faster delivery correlates with better ratings")
        print("   2. 📦 Improve food packaging to maintain temperature during delivery")
        print("   3. 🏪 Work with underperforming restaurants to improve food quality")
        print("   4. 📱 Enhance customer service response times for complaint resolution")
        print("   5. 🔄 Implement quality checks before dispatch to reduce wrong orders")
        
        return self.reviews_df
    
    def export_results(self, filename='swiggy_sentiment_analysis.csv'):
        """Export analysis results to CSV"""
        if self.reviews_df is None:
            raise ValueError("No reviews data found.")
        
        self.reviews_df.to_csv(filename, index=False)
        print(f"📁 Results exported to {filename}")
        
        # Also create a summary report
        summary_filename = 'swiggy_analysis_summary.txt'
        with open(summary_filename, 'w') as f:
            f.write("SWIGGY REVIEWS SENTIMENT ANALYSIS SUMMARY\n")
            f.write("="*50 + "\n\n")
            
            total_reviews = len(self.reviews_df)
            sentiment_counts = self.reviews_df['sentiment'].value_counts()
            
            f.write(f"Total Reviews Analyzed: {total_reviews}\n")
            f.write(f"Average Rating: {self.reviews_df['rating'].mean():.2f}/5\n")
            f.write(f"Average Delivery Time: {self.reviews_df['delivery_time'].mean():.1f} minutes\n\n")
            
            f.write("Sentiment Distribution:\n")
            for sentiment, count in sentiment_counts.items():
                percentage = (count / total_reviews) * 100
                f.write(f"  {sentiment}: {count} ({percentage:.1f}%)\n")
        
        print(f"📄 Summary report saved to {summary_filename}")

# Main execution
def main():
    """Main function to run the complete analysis"""
    print("🚀 Starting Swiggy Reviews Sentiment Analysis")
    print("="*50)
    
    # Initialize analyzer
    analyzer = SwiggyReviewAnalyzer()
    
    # Generate dummy reviews
    reviews_df = analyzer.generate_dummy_reviews(500)
    
    # Perform sentiment analysis
    analyzer.perform_sentiment_analysis()
    
    # Generate summary statistics
    analyzer.generate_summary_statistics()
    
    # Create visualizations
    analyzer.create_visualizations()
    
    # Analyze key insights
    analyzer.analyze_key_insights()
    
    # Export results
    analyzer.export_results()
    
    print(f"\n🎉 Analysis completed successfully!")
    print("📊 Check the generated visualizations and exported files for detailed insights.")
    
    return analyzer

# Run the analysis
if __name__ == "__main__":
    analyzer = main()
    
    # Optional: Display first few rows of analyzed data
    print("\n📋 Sample of analyzed data:")
    print(analyzer.reviews_df[['review_id', 'restaurant', 'rating', 'sentiment', 'polarity']].head(10))
