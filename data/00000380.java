public class StubbedTheories extends Theories { public StubbedTheories(Class<?> klass) throws InitializationError { super(klass); } @Override public Statement methodBlock(FrameworkMethod method) { return new StubbedTheoryAnchor(method, getTestClass()); } public static class StubbedTheoryAnchor extends TheoryAnchor { public StubbedTheoryAnchor(FrameworkMethod method, TestClass testClass) { super(method, testClass); } private List<GuesserQueue> queues = new ArrayList<GuesserQueue>(); @Override protected void handleAssumptionViolation(AssumptionViolatedException e) { super.handleAssumptionViolation(e); for (GuesserQueue queue : queues) { queue.update(e); } } @Override protected void runWithIncompleteAssignment(Assignments incomplete) throws Throwable { GuesserQueue guessers = createGuesserQueue(incomplete); queues.add(guessers); while (!guessers.isEmpty()) runWithAssignment(incomplete.assignNext(guessers.remove(0))); queues.remove(guessers); } private GuesserQueue createGuesserQueue(Assignments incomplete) throws Throwable { ParameterSignature nextUnassigned = incomplete.nextUnassigned(); if (nextUnassigned.hasAnnotation(Stub.class)) { GuesserQueue queue = new GuesserQueue(); queue.add(new Guesser<Object>(nextUnassigned.getType())); return queue; } return GuesserQueue.forSingleValues(incomplete.potentialsForNextUnassigned()); } } }