
````markdown
# Book Recommendation System using K-Nearest Neighbors

A machine learning-based book recommendation system that suggests similar books based on user ratings using the K-Nearest Neighbors algorithm.

## 📚 Project Overview

This project implements a collaborative filtering recommendation system using the Book-Crossings dataset. It analyzes user rating patterns to find and recommend books similar to a given title.

## 🎯 Features

- Recommends 5 similar books based on user rating patterns
- Uses K-Nearest Neighbors (KNN) algorithm with cosine similarity
- Filters data for statistical significance (active users and popular books)
- Returns similarity distances for each recommendation

## 📊 Dataset

**Book-Crossings Dataset:**
- 1.1 million ratings (scale of 1-10)
- 270,000 books
- 90,000 users

The dataset includes:
- `BX-Books.csv`: Book information (ISBN, title, author)
- `BX-Book-Ratings.csv`: User ratings (user ID, ISBN, rating)

## 🔧 Technologies Used

- **Python 3.x**
- **NumPy**: Numerical operations
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning (K-Nearest Neighbors)
- **SciPy**: Sparse matrix operations
- **Matplotlib**: Data visualization (optional)

## 🚀 Installation

```bash
# Install required libraries
pip install numpy pandas scipy scikit-learn matplotlib
````

## 💻 Usage

### Basic Usage

```python
# Get book recommendations
recommendations = get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))")
print(recommendations)
```

### Expected Output Format

```python
[
  'The Queen of the Damned (Vampire Chronicles (Paperback))',
  [
    ['Catch 22', 0.793983519077301], 
    ['The Witching Hour (Lives of the Mayfair Witches)', 0.7448656558990479], 
    ['Interview with the Vampire', 0.7345068454742432],
    ['The Tale of the Body Thief (Vampire Chronicles (Paperback))', 0.5376338362693787],
    ['The Vampire Lestat (Vampire Chronicles, Book II)', 0.5178412199020386]
  ]
]
```

The output contains:

1. The input book title
2. A list of 5 recommended books with their distances (lower = more similar)

## 🔍 How It Works

### 1. Data Preprocessing

- **Filter Active Users**: Remove users with fewer than 200 ratings
- **Filter Popular Books**: Remove books with fewer than 100 ratings
- **Merge Datasets**: Combine ratings with book information

### 2. Matrix Creation

- Create a user-book matrix (pivot table)
- Users as rows, books as columns
- Ratings as values, fill missing with 0
- Transpose to get book-user matrix

### 3. Model Training

- Convert matrix to sparse format for efficiency
- Train K-Nearest Neighbors model
    - **Metric**: Cosine similarity
    - **Algorithm**: Brute force
    - **Neighbors**: 6 (including the book itself)

### 4. Recommendation Generation

- Find the input book in the matrix
- Calculate distances to all other books
- Return 5 nearest neighbors (excluding the book itself)
- Sort by distance (furthest to closest)

## 📈 Algorithm Details

**K-Nearest Neighbors (KNN)**

- **Distance Metric**: Cosine distance
    - Measures the angle between rating vectors
    - Range: 0 (identical) to 1 (completely different)
- **Similarity Basis**: User rating patterns
    - Books rated similarly by the same users are considered similar

**Why Cosine Similarity?**

- Handles sparse data well
- Focuses on rating patterns rather than absolute values
- Effective for collaborative filtering

## 🎓 Code Structure

```python
# 1. Data Loading
df_books = pd.read_csv('BX-Books.csv', ...)
df_ratings = pd.read_csv('BX-Book-Ratings.csv', ...)

# 2. Data Filtering
# Filter users with >= 200 ratings
# Filter books with >= 100 ratings

# 3. Matrix Creation
user_book_matrix = df_merged.pivot_table(...)
book_user_matrix = user_book_matrix.T

# 4. Model Training
model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=6)
model.fit(matrix_sparse)

# 5. Recommendation Function
def get_recommends(book=""):
    # Returns [book_title, [[rec1, dist1], [rec2, dist2], ...]]
```

## ✅ Testing

The project includes a test function to validate recommendations:

```python
test_book_recommendation()
```

**Test Criteria:**

- Correct book title returned
- Expected books in recommendations
- Distance values within acceptable range (±0.05)

**Test Book:** "Where the Heart Is (Oprah's Book Club (Paperback))"

**Expected Recommendations:**

- I'll Be Seeing You
- The Weight of Water
- The Surgeon
- I Know This Much Is True

## 🎯 Key Challenges Solved

1. **Data Sparsity**: Most books aren't rated by most users
    
    - Solution: Filter for statistical significance
2. **Memory Efficiency**: Large user-book matrix
    
    - Solution: Use sparse matrix representation
3. **Computational Efficiency**: Distance calculations
    
    - Solution: Brute force algorithm works well with sparse data

## 📝 Notes

- The model uses collaborative filtering (user behavior patterns)
- Does not use content-based features (genre, author, description)
- Recommendations improve with more user rating data
- Distance values closer to 0 indicate higher similarity

## 🤝 Contributing

This is a learning project created as part of a data science challenge. Feel free to fork and experiment with:

- Different distance metrics (euclidean, manhattan)
- Varying filter thresholds
- Hybrid approaches (collaborative + content-based)

## 📄 License

Educational project - free to use and modify

## 🙏 Acknowledgments

- Dataset: Book-Crossings dataset
- Challenge: freeCodeCamp Machine Learning with Python certification

---

**Author:**  Software Developer specializing in Angular and Spring Boot  
**Focus:** Machine Learning, Recommendation Systems, Data Science

