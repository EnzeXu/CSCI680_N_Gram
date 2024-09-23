public class PublicClassValidatorTest { private final PublicClassValidator validator = new PublicClassValidator(); public static class PublicClass { } @Test public void acceptsPublicClass() { TestClass testClass = new TestClass(PublicClass.class); List<Exception> validationErrors = validator .validateTestClass(testClass); assertThat(validationErrors, is(equalTo(Collections.<Exception> emptyList()))); } static class NonPublicClass { } @Test public void rejectsNonPublicClass() { TestClass testClass = new TestClass(NonPublicClass.class); List<Exception> validationErrors = validator .validateTestClass(testClass); assertThat("Wrong number of errors.", validationErrors.size(), is(equalTo(1))); } }