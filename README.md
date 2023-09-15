# Asos-Plc-Trustpilot-Review-Analysis
This is the analysis of the customer reviews of Asos Plc that were posted on Trustpilot.co.uk (#1 trusted customer review site in the UK).


ANALYSIS OF ASOS TRUSTPILOT CUSTOMER REVIEW REPORT

INTRODUCTION

Sentiment analysis is a computational methodology that involves the identification of emotional nuances within comments, opinions, and textual data, subsequently classifying them into "Positive," "Negative," or "Neutral" categories. It has evolved into an indispensable tool for monitoring social media, serving to discern the general public's collective sentiments regarding specific topics, brands, products, or companies. This significance has been underscored by the substantial proliferation of user-generated content across social networking platforms, online reviews, and various text-based sources in recent years. Terminology associated with sentiment analysis can vary depending on the specific field of application. It exhibits connections to affective computing, which pertains to the computer's capacity to recognize and express emotions. In addition to its primary designation as sentiment analysis, it is alternatively referred to as subjectivity analysis, opinion mining, and evaluation extraction. In the realm of marketing, sentiment analysis has earned the moniker "Voice of Customers." This nomenclature underscores its pivotal role in furnishing organizations with invaluable insights into public sentiment, customer satisfaction, brand perception, and prevailing market trends. Consequently, sentiment analysis remains a cornerstone in the arsenal of tools employed by businesses to inform their strategic decisions and enhance their competitive edge.

The purpose of this report is to conduct a comprehensive analysis of the sentiments expressed by the public in Asos plc's customer reviews on Trustpilot.co.uk, which is the foremost review platform for products, brands, and companies in the United Kingdom. Trustpilot.co.uk holds a distinguished position as a social media website and is widely regarded as the most reputable review site in the UK. The primary objective of this sentiment analysis is to discern prevalent trends, detect patterns, and identify potential areas for enhancing the services offered by Asos plc. To accomplish this, we employed supervised machine learning techniques in conjunction with the Textblob Library, a Natural Language Processing ToolKit (NLTK) component. The customer reviews were categorized into one of three groups: positive, negative, or neutral, using sentiment scores assigned to words by Textblob. A comprehensive understanding of Asos customers' perception of the company, encompassing their experiences with products, brand interactions, customer support, delivery services, and return policies, is essential for informed business decision-making. This analysis serves as a valuable resource for gaining insights into customer satisfaction, recognizing emerging trends, safeguarding and enhancing brand reputation, and gauging the efficacy of marketing campaigns. In conclusion, the findings of this report are instrumental in providing Asos plc with actionable insights that can facilitate strategic improvements, reinforce customer relations, and maintain a competitive edge in the market.

Word Frequency Analysis: This analysis involves quantifying the frequency with which words appear in the reviews. It reveals the most commonly occurring words within the text, shedding light on the predominant themes and topics discussed. Word frequency analysis can be further dissected by categorizing words into different parts of speech, such as verbs, adverbs, and adjectives.

Word Cloud: The word cloud is a visualization tool designed to showcase the most frequently used words within a given dataset of text. This graphical representation offers an immediate and intuitive overview of the prominent terms. The size of each word in the word cloud corresponds to its frequency within the dataset. Additionally, word clouds can be customized to display general word frequency or specific word clouds for positive and negative sentiments, providing a swift and visually informative snapshot of popular terms associated with these sentiments.

Polarity Analysis: Polarity analysis gauges the degree of positivity or negativity expressed within a text. It quantifies the sentiment conveyed by a word, phrase, or entire text. The numerical measurement of text polarity typically ranges from -1 to 1, with negative polarity indicating a negative sentiment, positive polarity indicating a positive sentiment, and 0 representing a neutral sentiment.

Polarity Count: This metric signifies the tally of positive, negative, and neutral sentiments within an analyzed text. It serves as a valuable indicator of the proportion of positive, negative, or neutral opinions or reviews present in the text.

Subjectivity Analysis: Subjectivity analysis evaluates the extent to which factual, objective information contrasts with subjective, opinion-based content within the text. It quantifies the degree of subjectivity or objectivity present, with values ranging from 0 to 1. A score of 0 suggests a high level of objectivity, indicating that the text is primarily based on facts and devoid of personal opinions or emotions. Conversely, a score of 1 signifies a high level of subjectivity, indicating that the text is predominantly influenced by personal opinions.

Topic Modeling Analysis: Topic modelling employs Natural Language Processing algorithms to discern the principal themes and topics within a given text. It condenses extensive textual content into coherent categories, providing a structured overview of the core subjects discussed in the text. This analysis effectively categorizes reviews into distinct thematic areas.

Emotion Detection: Emotion detection encompasses the identification and quantification of various emotional states expressed within an analyzed text. It offers insights into the prevalence and distribution of emotions, such as fear, happiness, sadness, surprise, and anger. Unlike keyword-based analysis, emotion detection delves deeper into the text to capture context and nuances, ensuring a more accurate understanding of the emotions conveyed in the content.



METHODOLOGY

The sentiment analysis of customer reviews sourced from trustpilot.co.uk for Asos plc entails a series of significant steps and techniques. The ensuing section outlines the procedural framework for the analysis and categorization of these reviews.
I.	Data Collection and Preprocessing
Data Collection
The dataset utilized in this analysis comprises 3205 distinct reviews pertaining to Asos Plc, which were extracted from Trustpilot.co.uk. This dataset spans from May 15th to September 10th 2023, encompassing a comprehensive range of topics related to products and the customer experience. The data collection process involved the utilization of a third-party tool known as Octoparse to scrape reviews from the Trustpilot website, conducted through a keyword search using "ASOS" as the query term.
Data Preprocessing
The website offers a total of 12 categories of data that can be extracted. However, for the scope of this report, we narrowed down the extracted categories to six essential ones, including Star rating, Username, Location, Date, Review Topic, Reviews, and Date of Experience. These data were collected and stored in a CSV file, which was subsequently uploaded to Jupyter on Anaconda. Rigorous data cleaning procedures were conducted using Python to enhance the quality of the reviews. This involved removing URLs, emojis, and extraneous characters from the text, ensuring that we could perform a detailed analysis on clean, plain text.

Sentiment analysis in Python encompasses a variety of techniques designed to process text and categorize it based on the expressed sentiment. These techniques include:

•	Text Input: This initial step assumes the availability of preprocessed plain text ready for analysis.

•	Tokenization: This process involves breaking down words or text into smaller units for easier analysis.

•	Stop Word Filtering: Here, common words like "The," "a," "an," and "in" are disregarded by NLTK to focus on meaningful text during analysis.

•	Negation Handling: Negation detection and management are crucial for accurately interpreting text data that contains negations.

•	Stemming & Lemmatization: These techniques, integral to natural language processing (NLP), simplify complex words by reducing them to their root or base forms.

•	Classification: Text is assessed for sentiment, and labels such as positive, negative, or neutral are assigned based on the emotional content.

•	Sentiment Class: Refers to the category to which a text is assigned, i.e., positive, negative, or neutral.

It is noteworthy that all these techniques are seamlessly integrated into the Textblob library, automating much of the sentiment analysis process. Additionally, for the purposes of this analysis, reviews were filtered to include only those originating from the UK, resulting in a subset of 2,710 reviews out of the total of 3,205 scrapped from Trustpilot. This narrower focus allows for a more targeted examination of sentiment within the specified region.

II.	Sentiment Analysis Libraries

This analysis necessitates the utilization of several crucial libraries that prove instrumental at various stages of the analytical process. These essential libraries are outlined as follows:

•	Pandas Library: Employed for tasks such as reading, inspecting, merging, joining, filtering, and conducting statistical analyses on the dataset.

•	Matplotlib.pyplot Library: Utilized for crafting visualizations and generating polarity and subjectivity histograms.

•	Seaborn Library: Applied to create count plots for data visualization.

•	NumPy Library: Essential for numerical computations and analyses within the Python environment.

•	Textblob Library: Utilized to assess text sentiment by evaluating polarity and subjectivity.

•	Wordcloud Library: Facilitates the generation of word clouds, offering a comprehensive visual representation of frequently used words.

•	re (regular expression) Library: Employed to effectively eliminate superfluous elements such as emojis, URLs, and extraneous characters from the review text, streamlining it for analysis.

These libraries collectively empower the analysis by providing the necessary tools and functionality to conduct comprehensive and insightful assessments of the dataset.

III.	Data Analysis

The sentiment analysis of the dataset was performed using Python within a Jupyter Notebook environment. The corresponding .ipynb notebook will be provided as an attachment to facilitate thorough review and examination.


3. RESULTS AND DISCUSSION

1.	Word Frequency
   
 ![image](https://github.com/Mrtols/Asos-Plc-Trustpilot-Review-Analysis/assets/124041963/3e685a89-5137-41d0-acae-bf9f5a0a1a38)

Figure 3.1a: 30 Most common words in Asos Review
 ![image](https://github.com/Mrtols/Asos-Plc-Trustpilot-Review-Analysis/assets/124041963/f3f97f81-061a-44d4-95d3-e58ba9da2e4b)

Figure 3.1b: 30 Most common Verbs in Asos Review
 ![image](https://github.com/Mrtols/Asos-Plc-Trustpilot-Review-Analysis/assets/124041963/b8747b6d-0280-4b74-951a-845b70221f92)

Figure 3.1c: 30 Most Common Adverbs in Asos Review
![image](https://github.com/Mrtols/Asos-Plc-Trustpilot-Review-Analysis/assets/124041963/1592bd18-1c04-42bd-8902-26aebe5ddfa9)

Figure 3.1d: 30 Most Common Adjectives in Asos Review

Figure 3.1 (a, b, c, and d) presents an insightful analysis of the dataset, highlighting the prevalence of common words, verbs, adverbs, and adjectives. Notably, in Figure 3.1a, the two most frequently occurring words are "Customer" and "Service." This observation suggests that a substantial proportion of the reviews revolve around the customer service department. Additionally, the term "helpful" emerges prominently, indicating a positive sentiment regarding the effectiveness of the customer service team in resolving issues. It is worth noting that the majority of frequently used words in this context convey positive sentiments, with exceptions such as "issue" and "problem," which are to be expected and do not significantly detract from the overall positive tone of the reviews. Moving on to Figure 3.1b, which highlights the most prevalent action words, a predominantly positive inclination is evident. Words like "sorted," "resolved," and "helped" dominate this category, underscoring positive interactions and experiences. However, words like "Missing" and "Lost" introduce a contrasting negative sentiment, albeit to a lesser extent. Overall, this reinforces the notion that a substantial portion of the reviews convey positive sentiments. In Figure 3.1c and 3.1d, which delve into the most common adverbs and adjectives modifying verbs and nouns, respectively, a consistent trend of positivity emerges. These findings underscore the prevalence of positive remarks and commendable services provided by Asos Plc as reflected in the dataset. This analysis sheds light on the overall sentiment and prevailing themes within the reviews, emphasizing the predominantly positive nature of the customer feedback.
2.	Word Cloud
 ![image](https://github.com/Mrtols/Asos-Plc-Trustpilot-Review-Analysis/assets/124041963/436ea05d-d9c9-42d0-b62f-766989fa5403)

Figure 3.2a: General word cloud of Asos Customers Trustpilot Reviews

In Figure 3.2a, we gain valuable insights into the prevalent words within the entire set of reviews. The size of each word in this word cloud is directly proportional to its frequency of occurrence across the 2,667 reviews. Some words, such as "Customer," "Asos," "Service," "order", and "helpful," are self-evident due to their prominence. This visual representation provides a quick, at-a-glance overview of the most frequently mentioned terms, offering an effective means of comprehending popular topics within the dataset. To delve deeper into our analysis, we further segment this general word cloud into positive and negative word clouds, as detailed below.
Figure 3.2b narrows our focus to words with a notably high frequency of occurrence within reviews classified as having positive sentiments, as determined by Polarity Analysis. Among these conspicuous words are "Customer service," "Thank," "helpful," "Item," and "resolved." The positive polarity associated with these terms indicates a commendable level of satisfaction with the service provided by the customer service department. Furthermore, the presence of words such as "quick," "great," "friendly," "experience," and "return" implies that various aspects of the business operations and customer interactions appear to be meeting or exceeding expectations, contributing to overall satisfaction. This analysis provides a nuanced understanding of the prominent words within the reviews, emphasizing the positive sentiment and favourable experiences conveyed by customers in their feedback.
 ![image](https://github.com/Mrtols/Asos-Plc-Trustpilot-Review-Analysis/assets/124041963/7a031a8b-8542-449f-b221-150dad2a2a46)

Figure 3.2b: Positive WordCloud of Asos Customers Trustpilot Reviews
 ![image](https://github.com/Mrtols/Asos-Plc-Trustpilot-Review-Analysis/assets/124041963/1936ab4d-33bb-45d5-97ba-5b52c143a02e)

Figure 3.2c: Negative world cloud of Asos customer Trustpilot review

In Figure 3.2c, we shift our attention to words that exhibit a high frequency of occurrence within reviews characterized by negative polarity. It is paramount to closely examine negative reviews, as they offer valuable insights into areas requiring improvement. Several noteworthy terms, including "order," "next day," "delivery," and "customer service," appear to be associated with issues or challenges. For instance, the unit responsible for next-day delivery of orders appears to be encountering some operational difficulties, warranting an enhancement in their efficiency. Additionally, while the customer service department has shown satisfactory performance, it remains essential to further elevate their efficiency, considering that they serve as the initial point of contact for customers experiencing issues. Furthermore, the presence of terms like "Account" and "Refund" suggests specific concerns within the accounting department, indicating the need for streamlined and more efficient processes. This analysis underscores the significance of negative reviews in pinpointing areas where improvements are essential and provides valuable pointers for enhancing various operational aspects within Asos Plc.
3.	Polarity
 ![image](https://github.com/Mrtols/Asos-Plc-Trustpilot-Review-Analysis/assets/124041963/edbe0970-5fd8-4655-94f1-1f0c1cb45b75)

Figure 3.3: Polarity of Asos Customer Trustpilot Reviews

The histogram presented in Figure 3.3 provides a visual representation of the sentiment polarity distribution within the reviews. A noteworthy observation is that the majority of reviews exhibit a neutral polarity, suggesting that a substantial portion of individuals who have reviewed Asos Plc on Trust Pilot seem to be relatively content with the services provided. Moreover, it is evident that the number of positive reviews significantly outweighs the number of negative reviews, indicating a generally favorable public perception of Asos Plc. However, it is crucial to recognize that there is always room for improvement, as indicated by the presence of a minority of negative comments.

4.	Polarity Count
5.	
As illustrated in Figure 3.4, it offers a comprehensive view of sentiment distribution based on polarity, thereby enhancing our understanding of the overall sentiment landscape. Notably, out of the total reviews analyzed (approximately 2,254), around 1,400 reviews are classified as positive, which is a highly commendable reflection of the Asos brand. Negative reviews amount to approximately 400, with roughly 600 categorized as neutral. This distribution underscores the overall positive public perception of the company. However, it is essential to note that neutral reviews are often considered as a form of negative sentiment in the analysis. Therefore, a deeper examination of the neutral reviews is warranted to gain insight into their specific characteristics and any potential areas for improvement.
 ![image](https://github.com/Mrtols/Asos-Plc-Trustpilot-Review-Analysis/assets/124041963/ba872057-9144-4c9c-9867-826c2c53c92b)

Figure 3.4: The Sentiment Count of Asos Customers Trustpilot Reviews
 ![image](https://github.com/Mrtols/Asos-Plc-Trustpilot-Review-Analysis/assets/124041963/509ad360-6ad5-45cc-95f3-06b586ff2f39)

Figure 3.5: Subjectivity of Asos Customer’s Trustpilot reviews
In Figure 3.5, we examine the subjectivity of the reviews, revealing a combination of emotional content and factual information. Specifically, approximately 630 reviews are characterized by a high degree of objectivity, primarily rooted in factual content. Conversely, about 330 reviews exhibit a strong subjective element, predominantly driven by personal emotions and opinions. Notably, the majority of the remaining reviews, which constitute more than half of the dataset, incorporate a blend of both factual information and personal viewpoints, demonstrating a balanced mix of objective and subjective elements. This diversity in subjectivity levels within the reviews underscores the multifaceted nature of the feedback provided.

5.	Topic Modelling for Negative Reviews
 ![image](https://github.com/Mrtols/Asos-Plc-Trustpilot-Review-Analysis/assets/124041963/3e6d3796-6526-4712-96ff-e2039b9049d4)

Figure 3.6: Topic Modelling for Negative Reviews
In Figure 3.6, we gain valuable insights into the predominant topics that emerge within the negative reviews. This analysis provides a comprehensive overview of the key areas of concern expressed by reviewers. Topic #1 underscores the prominence of "Customer Service" as a central focus, with specific mention of issues related to product returns and the associated processing times. Additionally, the topic of "account" emerges, encompassing discussions related to both the account department, particularly in terms of refund processing and the Asos website's user accounts. The urgency surrounding "Next Day Delivery" and concerns pertaining to the company's "return policy" also surface prominently. These topics collectively highlight the areas of "Customer Service," "Account Department," "Next Day delivery," and "refund and return policies" as the central themes that dominate discussions within the negative reviews.

6.	Further Analysis of the Neutral Reviews (Topic Modelling)
 ![image](https://github.com/Mrtols/Asos-Plc-Trustpilot-Review-Analysis/assets/124041963/8c6c37e0-dea3-404c-8a5e-f3a003353a28)

Figure 3.7 Topic Modelling of the Neutral Reviews

In the context of sentiment analysis, when the number of neutral comments or reviews surpasses that of negative comments, it indicates the necessity for further investigation, as neutral comments often tend to be interpreted as leaning towards the negative end of the sentiment spectrum. To gain a deeper understanding of the content within these neutral comments, conducting a Topic Modeling Analysis is imperative. Figure 3.6 presents the outcomes of the topic modelling analysis applied to the neutral comments. Notably, Topics #1, #3, and #4 exhibit an absence of overtly negative terms, suggesting that these comments likely originate from individuals who had reasonably satisfactory experiences with Asos Plc. Furthermore, these topics prominently feature a higher frequency of positive words, indicating that they may even align more closely with positive reviews. However, Topic #2 introduces an element of negativity with terms such as "problem" and "issue." This suggests that these comments may belong to the subset of reviews typically categorized as negative. It is noteworthy that even within Topic #2, a mix of positive words is present, further emphasizing the complexity of sentiment within these neutral comments. This analysis underscores the significance of delving deeper into neutral comments to discern the nuances that may reveal positive or negative sentiments, thereby contributing to a more comprehensive understanding of customer feedback.

7.	Star Rating Analysis
Figure 3.7 presents a count plot illustrating the distribution of star ratings among the 2701 reviews. Impressively, approximately 1,550 reviews have garnered a 5-star rating, signifying a highly satisfactory and commendable response. Nevertheless, around 580 reviews convey varying degrees of dissatisfaction with one aspect of the service or another. This distribution underscores the inherent potential for improvement, as there remains ample room for enhancing the customer experience. Furthermore, it is worth noting that a modest number of reviews, less than 70 in total, reflect a high degree of satisfaction, while about 50 reviews indicate a lesser degree of satisfaction. This analysis underscores the fact that the majority of the neutral reviews actually tend to align more with positive sentiments, with only a minority exhibiting negative sentiments. In summary, the star rating distribution provides additional evidence that the neutral reviews predominantly lean towards the positive end of the sentiment spectrum, underscoring the need for a nuanced understanding of customer feedback.
 ![image](https://github.com/Mrtols/Asos-Plc-Trustpilot-Review-Analysis/assets/124041963/395a986e-fe72-4ee2-a2b9-e9a6e82f492f)

Figure 3.7: Asos customers Trustpilot Star Rating count plot

8.	Emotion Analysis
	
Figure 3.7 presents a count plot illustrating the distribution of star ratings among the 2,254 reviews. Impressively, approximately 1,550 reviews have garnered a 5-star rating, signifying a highly satisfactory and commendable response. Nevertheless, around 580 reviews convey varying degrees of dissatisfaction with one aspect of the service or another. This distribution underscores the inherent potential for improvement, as there remains ample room for enhancing the customer experience. Furthermore, it is worth noting that a modest number of reviews, less than 70 in total, reflect a high degree of satisfaction, while about 50 reviews indicate a lesser degree of satisfaction. This analysis underscores the fact that the majority of the neutral reviews actually tend to align more with positive sentiments, with only a minority exhibiting negative sentiments. In summary, the star rating distribution provides additional evidence that the neutral reviews predominantly lean towards the positive end of the sentiment spectrum, underscoring the need for a nuanced understanding of customer feedback.

 ![image](https://github.com/Mrtols/Asos-Plc-Trustpilot-Review-Analysis/assets/124041963/61a01805-ff85-4829-85f1-9454803e295d)

Figure 3.8a: Percentage of Emotion Analysis of Asos Reviews


4. BUSINESS INSIGHT
   
In the course of this sentiment analysis, a comprehensive examination of Asos Plc's Trustpilot customer reviews was conducted to uncover hidden trends and glean insights from this valuable dataset. Employing Python-based sentiment analysis methodologies and techniques, we probed into the brand perception of Asos Plc. From this rigorous analysis, the following insights have emerged:

•	Word Frequency Analysis underscores the significance of the customer service department, as its efficiency directly impacts the brand's overall perception. Given that this department serves as the initial point of contact for addressing issues or challenges with orders, improving its efficiency is crucial. Additionally, both customer care and the delivery unit should strive to enhance their effectiveness.

•	The analysis reveals that approximately 68% of the reviewed opinions of Asos Plc are highly positive, reflecting a strong degree of satisfaction. However, 25% express dissatisfaction with their experiences, while the remainder exhibit neutrality. This distribution emphasizes the effectiveness of the customer service but also underscores the room for continual improvement.

•	Examination of the reviews' Polarity and Subjectivity showcases that around 62% of opinions are characterized by positive sentiments, while 18% exhibit negativity, signifying areas that require enhancement. Most reviews display a blend of emotions, while 28% maintain a highly objective and positively polarized tone, indicating overall satisfaction. Conversely, 14% exhibit high subjectivity, likely reflecting negative sentiments.

•	A detailed analysis of neutral reviews using topic modeling demonstrates that a substantial portion of these reviews conveys positive sentiments, with a minority containing elements of negativity.

•	The topic modeling of negative reviews pinpoints specific areas within Asos service that demand immediate attention, such as Customer Service, Account Department, Next Day Delivery, and Refund and Return Policies.

•	An analysis of star ratings unveils that approximately 69% of reviews bestow a 5-star rating upon Asos. However, 25% of reviewers assign a 1-star rating, with the remaining 5% distributed among 2-star, 3-star, and 4-star ratings, with 4-star ratings being the most frequent at about 50 reviews.

•	The sentiment analysis uncovers notable trends, with the positive word cloud highlighting areas where the company excels, while the negative word cloud pinpoints aspects that require improvement, particularly within Customer Service, Next Day Delivery, order processing, and the accounting department.

•	Emotion Analysis categorizes reviews into various emotional states, with the majority expressing happiness, but about 22% reflecting fear, and 12.6% indicating sadness. The presence of sadness, fear, and anger underscores the imperative for enhancing service efficiency.

In summation, this sentiment analysis underscores the strong brand perception of "Asos Plc" based on customer reviews, particularly in the United Kingdom. However, it is evident that opportunities for improvement exist, notably within units like Customer Service, Next Day Delivery, order processing, and the accounting department, as indicated by negative reviews and the word cloud analysis. The efficient operation of these units should be a priority for further enhancing customer satisfaction and the brand's overall image.

6.4 LIMITATION AND CHALLENGES

Despite its utility in tracking brand mentions on social media, sentiment analysis is accompanied by several limitations and challenges that merit careful consideration:

Limitations

•	Sentiment analysis relies exclusively on data sourced from social media platforms, and not all customers utilize these platforms for providing feedback. Consequently, it should not be regarded as a perfectly accurate indicator of brand perception.

•	Natural Language Processing (NLP) used in sentiment analysis primarily processes plain text and phrases, which can lead to challenges in interpreting figurative language, such as irony and sarcasm. Additionally, machine learning algorithms may struggle to discern multiple context-dependent meanings for certain words.

•	The presence of diverse linguistic and cultural biases poses a significant challenge for sentiment analysis. NLP-based sentiment models may not account for cultural variations and different languages, resulting in erroneous conclusions when applied to various linguistic and cultural contexts.

•	The quality of the data under examination significantly impacts the accuracy of sentiment classification. Unstructured content, misspellings, abbreviations, and grammatical errors within the data can affect the precision of sentiment analysis results.

Challenges

•	Data sourcing presented a formidable challenge during this analysis. The initial plan to collect data from Twitter was hindered by changes in Twitter's leadership and algorithm, rendering traditional data scraping methods ineffective. This led to the adoption of Trustpilot as an alternative data source.

•	Data scraping from Trustpilot using Octoparse posed its own set of challenges. The Trustpilot website's algorithm restricted data scraping to a maximum of 280 rows at a time. Moreover, after scraping 280 rows on multiple occasions, subsequent attempts often yielded duplicate data from previous scrapes. Furthermore, data scraping on Trustpilot is limited to a monthly basis, which influenced the selection of data for analysis, limited to May and June 2023.

These limitations and challenges underscore the need for caution when employing sentiment analysis, emphasizing the importance of understanding its scope and potential inaccuracies in interpreting brand perception. Additionally, flexibility in data sourcing and recognition of the limitations of available sources are essential for conducting robust sentiment analyses.

6.5 RECOMMENDATION

The critical analysis of the reviews yields several key recommendations:

•	Prioritize Customer Service Efficiency: The primary focus should be on improving the efficiency of the customer service department. This can be achieved by addressing the heavy workload of customer inquiries, reducing response times, and enhancing overall efficiency. Potential strategies may include hiring additional customer service representatives, providing comprehensive training to existing staff to handle a wide range of inquiries effectively, or streamlining the customer service process.

•	Enhance Delivery Efficiency: Identify and rectify bottlenecks in the delivery process to ensure prompt and seamless order fulfillment. Consider optimizing tracking systems, emphasizing timely deliveries, and improving communication with customers regarding their orders.

•	Evaluate Refund and Return Policy: Given the mixed feedback in reviews, conduct a thorough evaluation of the refund and return policy to strike a balance that is fair to both customers and the company. Address concerns related to being overly strict or excessively lenient, aiming for a policy that is customer-centric and equitable.

•	Leverage Positive Customer Service Trends: Capitalize on the positive customer service trends highlighted in the reviews. Continue to provide exceptional service in areas identified by positive terms such as "Customer Service," "Sorted," and "Efficient." Building upon these strengths can further enhance the customer experience.

•	Address Varied Customer Concerns and Monitor Sentiments: Asos should proactively address a range of customer concerns by investigating and resolving issues as they arise. Additionally, establish a practice of ongoing sentiment monitoring to swiftly identify emerging concerns and implement proactive measures for their resolution.

These recommendations are derived from a comprehensive analysis of customer feedback and aim to improve various aspects of Asos's operations, ultimately leading to a more positive customer experience and brand perception.


CONCLUSION

Customer satisfaction is a pivotal factor contributing to the success of online fashion retailers such as Asos. The findings of this analysis, in conjunction with additional customer feedback, assume a crucial role in elevating the quality of services offered by Asos plc. Specifically, this analysis underscores the paramount importance of optimizing the performance of the Customer Service and Next-Day Delivery units, which significantly influence customer sentiment.

Furthermore, these insights are instrumental in facilitating well-informed decision-making and strategic planning. To harness the full potential for maximizing customer satisfaction, enhancing brand perception, and exploring new market opportunities, it is imperative that the Customer Service, Next-Day Delivery, and Accounting units implement key performance indicators (KPIs) as a means to drive continuous improvement initiatives.




