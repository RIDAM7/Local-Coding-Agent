from agent.models.schemas import ValidationReport

class FailureClassifier:
    @staticmethod
    def classify(report: ValidationReport) -> str:
        build_fail = report.build_result and not report.build_result.success
        lint_fail = report.lint_result and not report.lint_result.success
        test_fail = report.test_result and not report.test_result.success
        
        fails = sum([build_fail, lint_fail, test_fail])
        if fails == 0:
            return "NONE"
        elif fails > 1:
            return "MIXED_FAILURE"
        elif build_fail:
            return "BUILD_FAILURE"
        elif lint_fail:
            return "LINT_FAILURE"
        else:
            return "TEST_FAILURE"
