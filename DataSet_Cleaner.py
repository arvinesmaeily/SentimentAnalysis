# csv is used for reading and writing csv files and their rows
import csv
# Imported method is a natural language processor for English language
from spacy.lang.en import English
# Imported Set is a set of stop words which decrease performance of Sentiment Analysis
from spacy.lang.en.stop_words import STOP_WORDS


print("Step 1 of 3")

# Initialize an instance for English Natural Language Processor
nlp = English()

# Initialize a List of redundant symbols
filter_symbols = ['.', '..', '...', ',', '!', '?', '"', '\'', '@', '#', '$', '%', '^', '*', '(', ')', '-', '_', '=',
                  '+', '/', '\\', ';', ':', '[', ']', '{', '}', '|', ' ']

# Open given raw file in read-only mode
raw_input = open("Raw_DataFile.csv", 'r')

# Create and open a datafile for output writer of cleaned up dataset
clean_output = open("Clean_Datafile.csv", 'w', newline="")

print("Importing data in \"Raw_DataFile.csv\"...")

# Read raw datafile as csv and load it to raw_dataset object
raw_dataset = csv.reader(raw_input)

# Initialize a csv writer for clean dataset which gives the ability of
# writing desired rows into clean datafile
clean_dataset = csv.writer(clean_output)

print("Processing comments for stop words and characters...")
print("This process may take a while. Please wait...")

# Iteration through rows of raw dataset in order to filter redundant columns of data,
# clean up comment column, recombine new columns and write generated row into clean dataset
for row in raw_dataset:

    # first column of data is assigned to state column
    state = str(row[0])

    # second column of data is assigned to comment object
    # which is first tokenized by imported NLP
    comment = nlp(str(row[5]))

    # Initializing a list of tokens which contains tokens from comment column
    token_list = []
    for token in comment:
        token_list.append(token.text)

    # Initialize an empty filtered comment list
    filtered_comment = []

    # Iteration through tokens in token list, in order to assign the selected token
    # to lexeme if token is in vocab list in NLP, checking if lexeme is not a stop word or reduntant symbol
    # and appending it to the list of filtered comments
    for selected_token in token_list:
        lexeme = nlp.vocab[selected_token]
        if lexeme.is_stop == False and lexeme not in filter_symbols:
            filtered_comment.append(selected_token)

    clean_dataset.writerow([state, filtered_comment])

print("\"Clean_Datafile\" created successfully!")
