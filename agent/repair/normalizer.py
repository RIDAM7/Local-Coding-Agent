import re
from typing import List, Tuple
from agent.models.schemas import ValidationReport, StructuredDiagnostics, NormalizedDiagnostic
from agent.repair.classifier import FailureClassifier

class DiagnosticsNormalizer:
    @staticmethod
    def normalize(report: ValidationReport) -> Tuple[StructuredDiagnostics, NormalizedDiagnostic]:
        build_err = report.build_result.stderr if report.build_result and not report.build_result.success else ""
        if not build_err and report.build_result and not report.build_result.success:
             build_err = report.build_result.stdout
             
        lint_err = report.lint_result.stderr if report.lint_result and not report.lint_result.success else ""
        if not lint_err and report.lint_result and not report.lint_result.success:
             lint_err = report.lint_result.stdout
             
        test_err = report.test_result.stderr if report.test_result and not report.test_result.success else ""
        if not test_err and report.test_result and not report.test_result.success:
             test_err = report.test_result.stdout
             
        def truncate(s: str) -> str:
            if not s: return ""
            if len(s) > 3000:
                return s[:1500] + "\n...[TRUNCATED]...\n" + s[-1500:]
            return s
            
        classification = FailureClassifier.classify(report)
        
        primary_msg = ""
        combined_errs = ""
        if classification in ["BUILD_FAILURE", "MIXED_FAILURE"]:
            primary_msg = "Build failed. Fix compile/syntax errors first."
            combined_errs = build_err
        elif classification == "LINT_FAILURE":
            primary_msg = "Linting failed. Fix style or import errors."
            combined_errs = lint_err
        elif classification == "TEST_FAILURE":
            primary_msg = "Tests failed. Fix logical errors or regressions."
            combined_errs = test_err
            
        files = set()
        for ext in ['.py', '.js', '.ts', '.jsx', '.tsx']:
             matches = re.findall(r'([a-zA-Z0-9_/\-\.]+\\' + ext + r')', combined_errs)
             files.update(matches)
             
        struct = StructuredDiagnostics(
            build_errors=truncate(build_err),
            lint_errors=truncate(lint_err),
            test_errors=truncate(test_err),
            failed_files=list(files)
        )
        
        norm = NormalizedDiagnostic(
            classification=classification,
            primary_error_message=primary_msg,
            suspected_files=list(files)
        )
        
        return struct, norm
