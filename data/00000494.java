public class FrameworkMethodTest { @Rule public final ExpectedException thrown = none(); @Test public void cannotBeCreatedWithoutUnderlyingField() { thrown.expect(NullPointerException.class); thrown.expectMessage("FrameworkMethod cannot be created without an underlying method."); new FrameworkMethod(null); } @Test public void hasToStringWhichPrintsMethodName() throws Exception { Method method = ClassWithDummyMethod.class.getMethod("dummyMethod"); FrameworkMethod frameworkMethod = new FrameworkMethod(method); assertTrue(frameworkMethod.toString().contains("dummyMethod")); } @Test public void presentAnnotationIsAvailable() throws Exception { Method method = ClassWithDummyMethod.class.getMethod("annotatedDummyMethod"); FrameworkMethod frameworkMethod = new FrameworkMethod(method); Annotation annotation = frameworkMethod.getAnnotation(Rule.class); assertTrue(Rule.class.isAssignableFrom(annotation.getClass())); } @Test public void missingAnnotationIsNotAvailable() throws Exception { Method method = ClassWithDummyMethod.class.getMethod("annotatedDummyMethod"); FrameworkMethod frameworkMethod = new FrameworkMethod(method); Annotation annotation = frameworkMethod.getAnnotation(ClassRule.class); assertThat(annotation, is(nullValue())); } private static class ClassWithDummyMethod { @SuppressWarnings("unused") public void dummyMethod() { } @Rule public void annotatedDummyMethod() { } } }