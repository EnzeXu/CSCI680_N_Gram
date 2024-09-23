public class PrintableResult { private Result result; public static PrintableResult testResult(Class<?> type) { return testResult(Request.aClass(type)); } public static PrintableResult testResult(Request request) { return new PrintableResult(new JUnitCore().run(request)); } public PrintableResult(List<Failure> failures) { this(new FailureList(failures).result()); } private PrintableResult(Result result) { this.result = result; } public int failureCount() { return result.getFailures().size(); } public List<Failure> failures() { return result.getFailures(); } @Override public String toString() { ByteArrayOutputStream stream = new ByteArrayOutputStream(); new TextListener(new PrintStream(stream)).testRunFinished(result); return stream.toString(); } }