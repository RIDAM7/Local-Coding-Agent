from agent.review.schemas import ConfidenceReport, ReviewDecision

class ReviewRouter:
    def route(self, confidence_report: ConfidenceReport) -> ReviewDecision:
        if confidence_report.confidence_score >= 95.0:
            decision = ReviewDecision.APPROVE
        elif confidence_report.confidence_score >= 80.0:
            decision = ReviewDecision.REVIEW_REQUIRED
        else:
            decision = ReviewDecision.MANDATORY_REVIEW
            
        confidence_report.review_decision = decision
        return decision
