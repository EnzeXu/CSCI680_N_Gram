public class OrderingRequest extends MemoizingRequest { private final Request request; private final Ordering ordering; public OrderingRequest(Request request, Ordering ordering) { this.request = request; this.ordering = ordering; } @Override protected Runner createRunner() { Runner runner = request.getRunner(); try { ordering.apply(runner); } catch (InvalidOrderingException e) { return new ErrorReportingRunner(ordering.getClass(), e); } return runner; } }