public class FailOnTimeoutTest { @ Parameters ( name = "lookingForStuckThread = { 0 } " ) public static Iterable < Boolean > getParameters ( ) { return Arrays . asList ( Boolean . TRUE , Boolean . FALSE ) ; } @ Parameter public boolean lookingForStuckThread ; @ Test public void noExceptionIsThrownWhenWrappedStatementFinishesBeforeTimeoutWithoutThrowingException ( ) throws Throwable { FailOnTimeout failOnTimeout = failAfter50Ms ( new FastStatement ( ) ) ; failOnTimeout . evaluate ( ) ; } @ Test public void throwsTestTimedOutExceptionWithMeaningfulMessage ( ) { Exception e = assertThrows ( TestTimedOutException . class , run ( failAfter50Ms ( new RunForASecond ( ) ) ) ) ; assertEquals ( "test timed out after 50 milliseconds" , e . getMessage ( ) ) ; } @ Test public void sendUpExceptionThrownByStatement ( ) { Exception exception = new RuntimeException ( ) ; Exception e = assertThrows ( Exception . class , run ( failAfter50Ms ( new Fail ( exception ) ) ) ) ; assertSame ( exception , e ) ; } @ Test public void throwExceptionIfTheSecondCallToEvaluateNeedsTooMuchTime ( ) throws Throwable { DelegatingStatement statement = new DelegatingStatement ( ) ; FailOnTimeout failOnTimeout = failAfter50Ms ( statement ) ; statement . delegate = new FastStatement ( ) ; failOnTimeout . evaluate ( ) ; statement . delegate = new RunForASecond ( ) ; assertThrows ( TestTimedOutException . class , run ( failOnTimeout ) ) ; } @ Test public void throwTimeoutExceptionOnSecondCallAlthoughFirstCallThrowsException ( ) { DelegatingStatement statement = new DelegatingStatement ( ) ; FailOnTimeout failOnTimeout = failAfter50Ms ( statement ) ; statement . delegate = new Fail ( new AssertionError ( "first execution failed" ) ) ; assertThrows ( AssertionError . class , run ( failOnTimeout ) ) ; statement . delegate = new RunForASecond ( ) ; assertThrows ( TestTimedOutException . class , run ( failOnTimeout ) ) ; } @ Test public void throwsExceptionWithTimeoutValueAndTimeUnitSet ( ) { TestTimedOutException e = assertThrows ( TestTimedOutException . class , run ( failAfter50Ms ( new RunForASecond ( ) ) ) ) ; assertEquals ( 50 , e . getTimeout ( ) ) ; assertEquals ( MILLISECONDS , e . getTimeUnit ( ) ) ; } @ Test public void statementThatCanBeInterruptedIsStoppedAfterTimeout ( ) throws Throwable { RunForASecond runForASecond = new RunForASecond ( ) ; assertThrows ( TestTimedOutException . class , run ( failAfter50Ms ( runForASecond ) ) ) ; boolean stopped = runForASecond . finished . await ( 50 , MILLISECONDS ) ; assertTrue ( "Thread has not been stopped . " , stopped ) ; } @ Test public void stackTraceContainsRealCauseOfTimeout ( ) { TestTimedOutException timedOutException = assertThrows ( TestTimedOutException . class , run ( failAfter50Ms ( new StuckStatement ( ) ) ) ) ; StackTraceElement [ ] stackTrace = timedOutException . getStackTrace ( ) ; boolean stackTraceContainsTheRealCauseOfTheTimeout = false ; boolean stackTraceContainsOtherThanTheRealCauseOfTheTimeout = false ; for ( StackTraceElement element : stackTrace ) { String methodName = element . getMethodName ( ) ; if ( "theRealCauseOfTheTimeout" . equals ( methodName ) ) { stackTraceContainsTheRealCauseOfTheTimeout = true ; } if ( "notTheRealCauseOfTheTimeout" . equals ( methodName ) ) { stackTraceContainsOtherThanTheRealCauseOfTheTimeout = true ; } } assertTrue ( "Stack trace does not contain the real cause of the timeout" , stackTraceContainsTheRealCauseOfTheTimeout ) ; assertFalse ( "Stack trace contains other than the real cause of the timeout , which can be very misleading" , stackTraceContainsOtherThanTheRealCauseOfTheTimeout ) ; } private static final class StuckStatement extends Statement { @ Override public void evaluate ( ) throws Throwable { try { theRealCauseOfTheTimeout ( ) ; } catch ( InterruptedException e ) { } finally { notTheRealCauseOfTheTimeout ( ) ; } } private void theRealCauseOfTheTimeout ( ) throws InterruptedException { sleep ( MAX_VALUE ) ; } private void notTheRealCauseOfTheTimeout ( ) { for ( long now = currentTimeMillis ( ) , eta = now + 1000L ; now < eta ; now = currentTimeMillis ( ) ) { atan ( now ) ; } } } @ Test public void lookingForStuckThread_threadGroupNotLeaked ( ) throws Throwable { assumeTrue ( lookingForStuckThread ) ; assumeTrue ( "Thread groups can no longer be destroyed on JDK 16 and later" , System . getProperty ( "java . vm . specification . version" ) . compareTo ( "16" ) < 0 ) ; final AtomicReference < ThreadGroup > innerThreadGroup = new AtomicReference < ThreadGroup > ( ) ; final AtomicReference < Thread > innerThread = new AtomicReference < Thread > ( ) ; final ThreadGroup outerThreadGroup = currentThread ( ) . getThreadGroup ( ) ; FailOnTimeout failOnTimeout = failAfter50Ms ( new Statement ( ) { @ Override public void evaluate ( ) { innerThread . set ( currentThread ( ) ) ; ThreadGroup group = currentThread ( ) . getThreadGroup ( ) ; assertNotSame ( "inner thread should use a different thread group" , outerThreadGroup , group ) ; innerThreadGroup . set ( group ) ; assertTrue ( "the 'FailOnTimeoutGroup' thread group should be a daemon thread group" , group . isDaemon ( ) ) ; } } ) ; failOnTimeout . evaluate ( ) ; assertNotNull ( "the Statement was never run" , innerThread . get ( ) ) ; innerThread . get ( ) . join ( ) ; assertTrue ( "the 'FailOnTimeoutGroup' thread group should be destroyed after running the test" , innerThreadGroup . get ( ) . isDestroyed ( ) ) ; } @ Test public void notLookingForStuckThread_usesSameThreadGroup ( ) throws Throwable { assumeFalse ( lookingForStuckThread ) ; final AtomicBoolean statementWasExecuted = new AtomicBoolean ( ) ; final ThreadGroup outerThreadGroup = currentThread ( ) . getThreadGroup ( ) ; FailOnTimeout failOnTimeout = failAfter50Ms ( new Statement ( ) { @ Override public void evaluate ( ) { statementWasExecuted . set ( true ) ; ThreadGroup group = currentThread ( ) . getThreadGroup ( ) ; assertSame ( "inner thread should use the same thread group" , outerThreadGroup , group ) ; } } ) ; failOnTimeout . evaluate ( ) ; assertTrue ( "the Statement was never run" , statementWasExecuted . get ( ) ) ; } private FailOnTimeout failAfter50Ms ( Statement statement ) { return FailOnTimeout . builder ( ) . withTimeout ( 50 , MILLISECONDS ) . withLookingForStuckThread ( lookingForStuckThread ) . build ( statement ) ; } private ThrowingRunnable run ( final FailOnTimeout failOnTimeout ) { return new ThrowingRunnable ( ) { public void run ( ) throws Throwable { failOnTimeout . evaluate ( ) ; } } ; } private static class DelegatingStatement extends Statement { volatile Statement delegate ; @ Override public void evaluate ( ) throws Throwable { delegate . evaluate ( ) ; } } private static class FastStatement extends Statement { @ Override public void evaluate ( ) throws Throwable { } } private static final class RunForASecond extends Statement { final CountDownLatch finished = new CountDownLatch ( 1 ) ; @ Override public void evaluate ( ) throws Throwable { long timeout = currentTimeMillis ( ) + 1000L ; while ( !interrupted ( ) && currentTimeMillis ( ) < timeout ) { } finished . countDown ( ) ; } } }