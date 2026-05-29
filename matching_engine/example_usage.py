"""
Example: AI Matching Engine Usage

Demonstrates the full 6-stage pipeline matching resumes against a JD.
"""

import asyncio
import logging

from matching_engine.pipeline import MatchingPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

# Sample Job Description
SAMPLE_JD = """
Senior Backend Developer - Java/Microservices

We are looking for a Senior Backend Developer to join our platform team.

Requirements:
- 5-8 years of experience in backend development
- Strong proficiency in Java, Spring Boot, and Microservices architecture
- Experience with AWS (EC2, Lambda, S3, DynamoDB)
- Hands-on experience with Kubernetes and Docker
- Proficiency in SQL and NoSQL databases
- Experience with CI/CD pipelines and DevOps practices
- Strong understanding of RESTful APIs and event-driven architecture

Good to have:
- Experience with Kafka or RabbitMQ
- Knowledge of React or Angular for full-stack contributions
- AWS certifications (Solutions Architect, Developer)
- Experience in fintech or e-commerce domain

Education: Bachelor's degree in Computer Science or related field
Location: Bangalore, India
"""

# Sample Resumes
SAMPLE_RESUME_1 = """
Rohit Kumar Sharma
Email: rohit.sharma@email.com | Phone: 9876543210
Location: Bangalore, India

Career Summary:
Senior Software Engineer with 6.8 years of experience specializing in Java backend
development and cloud-native microservices. Strong track record of building scalable
distributed systems on AWS.

Skills:
Java, Spring Boot, Microservices, AWS (EC2, Lambda, S3, DynamoDB), Docker, Kubernetes,
PostgreSQL, MongoDB, Redis, Kafka, REST APIs, CI/CD, Jenkins, Git, Agile

Work Experience:
1. Senior Software Engineer | TechCorp Solutions | 2021 - Present
   - Designed and built microservices handling 10M+ daily transactions
   - Led migration from monolith to microservices on AWS EKS
   - Technologies: Java, Spring Boot, AWS, Kubernetes, Kafka, PostgreSQL

2. Software Engineer | CloudFirst Inc | 2018 - 2021
   - Built RESTful APIs serving 50+ internal and external consumers
   - Implemented CI/CD pipelines using Jenkins and Docker
   - Technologies: Java, Spring Boot, Docker, MySQL, Redis

Projects:
- Payment Gateway Service: High-throughput payment processing microservice (Java, Spring Boot, AWS, Kafka)
- Real-time Analytics Platform: Event-driven data pipeline (Kafka, Spring Cloud Stream, DynamoDB)

Education: B.Tech Computer Science - IIT Delhi (2017)
Certifications: AWS Solutions Architect Associate
"""

SAMPLE_RESUME_2 = """
Anita M. Iyer
Email: anita.iyer@email.com | Phone: 9123456780
Location: Mumbai, India

Career Summary:
Software Developer with 5.4 years of experience in Java and cloud technologies.
Focused on building reliable backend services with strong database skills.

Skills:
Java, Spring Boot, AWS (EC2, S3), SQL, Docker, REST APIs, MySQL, PostgreSQL,
Git, Maven, JUnit, Agile/Scrum

Work Experience:
1. Software Developer | DataSystems Pvt Ltd | 2020 - Present
   - Developed REST APIs for customer management platform
   - Managed MySQL and PostgreSQL databases (100M+ records)
   - Technologies: Java, Spring Boot, AWS EC2, MySQL, Docker

2. Junior Developer | WebTech Solutions | 2019 - 2020
   - Built backend modules for e-commerce platform
   - Technologies: Java, Spring MVC, MySQL, Git

Projects:
- Inventory Management System: Backend service for warehouse management (Java, Spring Boot, MySQL)
- Customer Analytics Dashboard: Data aggregation service (Java, PostgreSQL, REST APIs)

Education: M.Tech Software Engineering - BITS Pilani (2019)
"""

SAMPLE_RESUME_3 = """
Vikram Reddy
Email: vikram.reddy@email.com | Phone: 9988776655
Location: Hyderabad, India

Career Summary:
Experienced backend developer with 7.2 years building enterprise Java applications.
Strong in Spring ecosystem and database technologies with recent Kafka experience.

Skills:
Java, Spring Boot, Spring Cloud, Microservices, SQL, PostgreSQL, Oracle, Kafka,
Docker, Maven, Gradle, JUnit, Mockito, REST APIs, GraphQL, Git

Work Experience:
1. Lead Developer | Enterprise Solutions Ltd | 2020 - Present
   - Leading team of 5 developers building microservices platform
   - Architected event-driven system using Kafka
   - Technologies: Java, Spring Boot, Spring Cloud, Kafka, PostgreSQL, Docker

2. Senior Developer | FinServ Technologies | 2017 - 2020
   - Built core banking microservices handling financial transactions
   - Technologies: Java, Spring Boot, Oracle, REST APIs, Docker

Projects:
- Trade Settlement Engine: Real-time trade processing (Java, Spring Boot, Kafka, PostgreSQL)
- API Gateway: Custom API gateway with rate limiting (Spring Cloud Gateway, Redis)

Education: B.E. Computer Science - JNTU Hyderabad (2017)
Certifications: Oracle Certified Java Professional
"""


async def main():
    """Run the matching pipeline with sample data."""
    pipeline = MatchingPipeline(
        model="ollama/llama2",
        embedding_model="all-MiniLM-L6-v2",
    )

    print("=" * 70)
    print("AI MATCHING ENGINE - Resume vs JD Matching")
    print("=" * 70)

    results = await pipeline.match(
        jd_text=SAMPLE_JD,
        resume_texts=[SAMPLE_RESUME_1, SAMPLE_RESUME_2, SAMPLE_RESUME_3],
    )

    # Display results (Candidate Match Grid)
    print("\n" + "=" * 70)
    print("RESULTS - Candidate Match Grid")
    print("=" * 70)
    print(f"{'Name':<25} {'Exp (Yrs)':<12} {'Match %':<10} {'Recommendation'}")
    print("-" * 70)

    for result in results:
        name = result.candidate.full_name
        exp = result.candidate.total_experience_years or "N/A"
        score = f"{result.qualification_percentage}%"
        rec = result.recommendation
        print(f"{name:<25} {str(exp):<12} {score:<10} {rec}")

    # Detailed view for top candidate
    if results:
        top = results[0]
        print("\n" + "=" * 70)
        print("CANDIDATE DETAIL VIEW - Top Match")
        print("=" * 70)
        print(f"Name: {top.candidate.full_name}")
        print(f"Phone: {top.candidate.phone}")
        print(f"Email: {top.candidate.email}")
        print(f"Qualification: {top.qualification_percentage}%")
        print(f"\nAI Reasoning:")
        print(f"  {top.reasoning}")
        print(f"\nMatched Strengths:")
        for s in top.key_strengths:
            print(f"  + {s}")
        print(f"\nMissing / Gap Areas:")
        for s in top.missing_skills:
            print(f"  - {s}")
        print(f"\nRecommendation:")
        print(f"  {top.recommendation}")

        # Scoring breakdown
        print(f"\nScoring Breakdown:")
        sb = top.scoring_breakdown
        print(f"  Must-have match:    {sb.must_have_match:.0%}")
        print(f"  Experience match:   {sb.experience_match:.0%}")
        print(f"  Skills depth:       {sb.skills_depth:.0%}")
        print(f"  Project relevance:  {sb.project_relevance:.0%}")
        print(f"  Recency factor:     {sb.recency_factor:.0%}")


if __name__ == "__main__":
    asyncio.run(main())
