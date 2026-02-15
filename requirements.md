# Requirements Document: Kirana AI Advisor

## Introduction

The Kirana AI Advisor is an AI-powered management system designed to transform traditional Indian Kirana stores from manual operators into data-driven strategists. Unlike existing solutions that focus on digital ledgers (MargBooks, JioMart Partner), this system provides proactive advisory through deep data analysis, hybrid intelligence integrating wholesale Mandi prices and local festival calendars, and future-looking predictions. The system leverages RAG (Retrieval-Augmented Generation) and Agentic AI to deliver precision inventory management, demand prediction, and real-time supply-demand inflation detection.

## Glossary

- **Kirana_Store**: Traditional neighborhood retail store in India selling groceries and daily essentials
- **RAG_System**: Retrieval-Augmented Generation system combining vector database retrieval with LLM generation
- **Mandi**: Wholesale agricultural market in India where farmers sell produce
- **Udhaar**: Credit system where customers buy goods on credit to be paid later
- **Hinglish**: Hybrid language mixing Hindi and English commonly used in India
- **Khatabook**: Physical or digital ledger book for recording transactions
- **Agmarknet**: Government portal providing agricultural market prices
- **Neuro_Symbolic_Engine**: AI architecture combining neural networks with symbolic reasoning
- **Logic_Library**: Rule-based system encoding Indian retail cultural nuances
- **Demand_Predictor**: ML model forecasting product demand based on historical and contextual data
- **Inflation_Detector**: Real-time monitoring system tracking supply-demand price changes
- **OCR_Engine**: Optical Character Recognition system for extracting text from images
- **Vector_Database**: Database storing embeddings for semantic search (ChromaDB/Pinecone)
- **Chat_Interface**: Conversational UI supporting natural language queries
- **Mobile_App**: Android application built with Kotlin and Jetpack Compose
- **Backend_API**: FastAPI/Flask server handling business logic and AI inference
- **Data_Scraper**: Automated system collecting market data from government portals
- **Festival_Calendar**: Database of Indian festivals and regional celebrations
- **Outlier_Handler**: System detecting and managing exceptional events (weddings, bulk orders)

## Requirements

### Requirement 1: Data Ingestion and Processing

**User Story:** As a Kirana store owner, I want to upload my transaction data through Excel files or capture Khatabook images, so that the system can analyze my business without manual data entry.

#### Acceptance Criteria

1. WHEN a user uploads an Excel file (.csv or .xlsx), THE Data_Ingestion_System SHALL parse the file and extract transaction records within 10 seconds
2. WHEN a user captures a Khatabook image, THE OCR_Engine SHALL extract text with at least 90% accuracy for printed Hindi and English text
3. WHEN OCR extraction completes, THE Data_Ingestion_System SHALL validate extracted data against expected schema and flag inconsistencies
4. WHEN data validation succeeds, THE Data_Ingestion_System SHALL store records in MySQL database with proper indexing
5. IF data validation fails, THEN THE Data_Ingestion_System SHALL return specific error messages indicating which fields are invalid
6. WHEN processing handwritten Khatabook images, THE OCR_Engine SHALL achieve at least 75% accuracy for common Hindi and English words
7. THE Data_Ingestion_System SHALL support batch uploads of up to 1000 transaction records in a single operation
8. WHEN duplicate transactions are detected, THE Data_Ingestion_System SHALL prompt the user for confirmation before storing

### Requirement 2: Multilingual Conversational Interface

**User Story:** As a Kirana store owner who speaks Hinglish or vernacular languages, I want to ask questions in my natural language, so that I can get business insights without learning technical terms.

#### Acceptance Criteria

1. WHEN a user sends a query in Hinglish, THE Chat_Interface SHALL process and respond appropriately within 5 seconds
2. WHEN a user asks "How much rice for next week?", THE RAG_System SHALL retrieve relevant historical data and provide demand prediction
3. THE Chat_Interface SHALL support Hindi, English, and Hinglish input with automatic language detection
4. WHEN a user switches languages mid-conversation, THE Chat_Interface SHALL maintain context and respond in the new language
5. THE Chat_Interface SHALL recognize domain-specific terms like "Udhaar", "Mandi bhav", "festival stock" without requiring translation
6. WHEN ambiguous queries are received, THE Chat_Interface SHALL ask clarifying questions before providing answers
7. THE Chat_Interface SHALL maintain conversation history for context-aware responses across multiple turns
8. WHEN voice input is provided, THE Mobile_App SHALL transcribe speech to text with support for Indian English accents

### Requirement 3: Demand Prediction and Inventory Optimization

**User Story:** As a Kirana store owner, I want AI-powered demand predictions for my products, so that I can optimize inventory and reduce waste or stockouts.

#### Acceptance Criteria

1. WHEN a user requests demand forecast for a product, THE Demand_Predictor SHALL generate predictions for the next 7, 14, and 30 days
2. THE Demand_Predictor SHALL incorporate historical sales data, seasonal patterns, and upcoming festivals in predictions
3. WHEN a major festival is within 14 days, THE Demand_Predictor SHALL increase predicted demand for festival-related products by appropriate factors
4. THE Demand_Predictor SHALL achieve at least 75% accuracy (MAPE < 25%) for top 20 products over 30-day periods
5. WHEN inventory levels fall below predicted demand threshold, THE System SHALL generate proactive restocking alerts
6. THE Demand_Predictor SHALL identify slow-moving products with less than 2 units sold per month
7. WHEN outlier events are detected (bulk orders, weddings), THE Outlier_Handler SHALL exclude them from baseline demand calculations
8. THE Demand_Predictor SHALL provide confidence intervals for predictions to indicate uncertainty levels

### Requirement 4: Real-Time Market Intelligence Integration

**User Story:** As a Kirana store owner, I want real-time wholesale Mandi prices and supply-demand trends, so that I can make informed purchasing decisions and adjust pricing.

#### Acceptance Criteria

1. THE Data_Scraper SHALL fetch Mandi prices from Agmarknet daily at 6 AM IST
2. WHEN Mandi prices change by more than 15% for any product, THE Inflation_Detector SHALL send immediate alerts to users
3. THE System SHALL maintain a 90-day rolling history of Mandi prices for trend analysis
4. WHEN a user queries current market prices, THE System SHALL display prices from nearest 3 Mandis with timestamps
5. THE Inflation_Detector SHALL calculate supply-demand inflation indicators using price velocity and volume data
6. WHEN supply shortages are detected (price increase + volume decrease), THE System SHALL recommend alternative suppliers or products
7. THE Data_Scraper SHALL handle government portal downtime gracefully and retry with exponential backoff
8. THE System SHALL scrape GST and regulatory updates from government portals weekly and notify users of relevant changes

### Requirement 5: Neuro-Symbolic Reasoning Engine

**User Story:** As a system architect, I want a Neuro-Symbolic architecture combining AI with business logic, so that the system provides culturally-aware and theoretically-sound recommendations.

#### Acceptance Criteria

1. THE Neuro_Symbolic_Engine SHALL combine neural network predictions with symbolic rules from the Logic_Library
2. THE Logic_Library SHALL encode rules for Udhaar credit lifecycle including credit limits, payment reminders, and default risk
3. WHEN calculating credit limits, THE Neuro_Symbolic_Engine SHALL apply both ML-based customer scoring and rule-based constraints
4. THE Logic_Library SHALL include seasonal rules for Indian festivals (Diwali, Holi, Eid, Pongal, Onam) with regional variations
5. WHEN contradictions arise between neural predictions and symbolic rules, THE Neuro_Symbolic_Engine SHALL prioritize symbolic rules and log the conflict
6. THE Logic_Library SHALL support dynamic rule updates without requiring model retraining
7. THE Neuro_Symbolic_Engine SHALL provide explainable reasoning chains showing both data-driven and rule-based components
8. WHEN processing business queries, THE Neuro_Symbolic_Engine SHALL complete inference within 5 seconds using Chain of Thought reasoning

### Requirement 6: Cultural and Seasonal Intelligence

**User Story:** As a Kirana store owner, I want the system to understand Indian retail culture and seasonal patterns, so that recommendations align with local business practices.

#### Acceptance Criteria

1. THE Festival_Calendar SHALL include national and regional festivals with dates for the next 2 years
2. WHEN a festival approaches, THE System SHALL automatically suggest relevant product categories for stocking
3. THE System SHALL recognize regional variations (North vs South India) in festival celebrations and product preferences
4. THE Logic_Library SHALL model Udhaar credit patterns including typical credit periods (7, 15, 30 days) and collection strategies
5. WHEN analyzing customer behavior, THE System SHALL distinguish between regular customers, credit customers, and one-time buyers
6. THE System SHALL provide recommendations for festival-specific pricing strategies and promotional bundles
7. THE System SHALL track wedding season patterns and adjust demand predictions for bulk orders accordingly
8. WHEN unusual bulk orders occur, THE Outlier_Handler SHALL flag them separately to prevent skewing baseline demand models

### Requirement 7: Mobile Application Interface

**User Story:** As a Kirana store owner, I want a mobile app with intuitive UI, so that I can manage my store on-the-go without technical expertise.

#### Acceptance Criteria

1. THE Mobile_App SHALL provide a dashboard showing daily sales, inventory alerts, and top recommendations
2. WHEN the app launches, THE Mobile_App SHALL display critical alerts (stockouts, price changes) within 3 seconds
3. THE Mobile_App SHALL support offline mode for viewing cached data and recording transactions
4. WHEN internet connectivity is restored, THE Mobile_App SHALL sync offline transactions to the server automatically
5. THE Mobile_App SHALL integrate device camera for Khatabook image capture with real-time preview
6. THE Mobile_App SHALL provide voice input for chat queries using device microphone
7. THE Mobile_App SHALL display data visualizations (charts, graphs) for sales trends and inventory levels
8. THE Mobile_App SHALL support dark mode and adjustable font sizes for accessibility

### Requirement 8: Backend API and AI Inference

**User Story:** As a system developer, I want a scalable backend API with fast AI inference, so that the system can serve multiple users with low latency.

#### Acceptance Criteria

1. THE Backend_API SHALL handle at least 100 concurrent users with average response time under 2 seconds
2. WHEN AI inference is requested, THE Backend_API SHALL return results within 5 seconds for 95% of queries
3. THE Backend_API SHALL use fine-tuned Llama 3.1 8B model for natural language understanding and generation
4. THE Backend_API SHALL implement rate limiting to prevent abuse (100 requests per user per hour)
5. THE Backend_API SHALL provide RESTful endpoints for all core operations (data upload, queries, predictions)
6. WHEN errors occur, THE Backend_API SHALL return structured error responses with appropriate HTTP status codes
7. THE Backend_API SHALL log all requests and responses for debugging and analytics
8. THE Backend_API SHALL implement authentication using JWT tokens with 24-hour expiration

### Requirement 9: Vector Database and RAG System

**User Story:** As a system architect, I want a RAG system with vector database, so that the AI can retrieve relevant context for accurate and grounded responses.

#### Acceptance Criteria

1. THE Vector_Database SHALL store embeddings for transaction history, product catalog, and market intelligence
2. WHEN a user query is received, THE RAG_System SHALL retrieve top 5 most relevant documents based on semantic similarity
3. THE RAG_System SHALL combine retrieved context with LLM generation to produce grounded responses
4. THE Vector_Database SHALL update embeddings incrementally as new data is ingested
5. THE RAG_System SHALL achieve retrieval precision of at least 80% for domain-specific queries
6. WHEN no relevant context is found, THE RAG_System SHALL inform the user rather than hallucinating information
7. THE Vector_Database SHALL support filtering by date range, product category, and store location
8. THE RAG_System SHALL use ChromaDB or Pinecone for vector storage with automatic scaling

### Requirement 10: Data Security and Privacy

**User Story:** As a Kirana store owner, I want my business data to be secure and private, so that my competitive information is protected.

#### Acceptance Criteria

1. THE System SHALL encrypt all data in transit using TLS 1.3
2. THE System SHALL encrypt sensitive data at rest (customer information, financial records) using AES-256
3. THE System SHALL implement role-based access control with separate permissions for owners, staff, and auditors
4. WHEN a user logs in, THE System SHALL require multi-factor authentication for first-time devices
5. THE System SHALL comply with Indian data protection regulations and store data within India
6. THE System SHALL provide data export functionality allowing users to download their complete data
7. THE System SHALL implement automatic session timeout after 30 minutes of inactivity
8. WHEN suspicious activity is detected, THE System SHALL lock the account and notify the owner

### Requirement 11: Analytics and Reporting

**User Story:** As a Kirana store owner, I want comprehensive analytics and reports, so that I can understand my business performance and make strategic decisions.

#### Acceptance Criteria

1. THE System SHALL generate daily, weekly, and monthly sales reports with revenue breakdowns
2. THE System SHALL provide profit margin analysis by product category and individual SKU
3. THE System SHALL track customer purchase patterns and identify top customers by revenue
4. THE System SHALL calculate inventory turnover ratios and highlight slow-moving stock
5. THE System SHALL provide Udhaar credit reports showing outstanding amounts, aging, and collection rates
6. THE System SHALL compare actual sales against predicted demand to measure forecast accuracy
7. THE System SHALL generate year-over-year comparison reports for seasonal analysis
8. WHEN reports are generated, THE System SHALL allow export in PDF and Excel formats

### Requirement 12: System Monitoring and Reliability

**User Story:** As a system administrator, I want comprehensive monitoring and error handling, so that the system remains reliable and issues are quickly identified.

#### Acceptance Criteria

1. THE System SHALL achieve 99.5% uptime measured monthly
2. THE System SHALL implement health check endpoints for all critical services
3. WHEN any service fails, THE System SHALL automatically restart and log the failure
4. THE System SHALL monitor API response times and alert when 95th percentile exceeds 5 seconds
5. THE System SHALL implement circuit breakers for external dependencies (Agmarknet, OCR services)
6. THE System SHALL maintain audit logs for all data modifications with user attribution
7. THE System SHALL perform automated database backups daily with 30-day retention
8. WHEN database queries exceed 10 seconds, THE System SHALL log slow queries for optimization
