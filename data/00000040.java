public class DummyListenableFuture < T > extends AbstractListenableFuture < T , GenericCompletionListener > { private boolean done ; private boolean cancelled = false ; private T content = null ; public DummyListenableFuture ( boolean alreadyDone , ExecutorService service ) { super ( service ) ; this . done = alreadyDone ; } @ Override public boolean cancel ( boolean bln ) { cancelled = true ; notifyListeners ( ) ; return true ; } @ Override public boolean isCancelled ( ) { return cancelled ; } @ Override public boolean isDone ( ) { return done ; } @ Override public T get ( ) throws InterruptedException , ExecutionException { try { return get ( 1 , TimeUnit . SECONDS ) ; } catch ( TimeoutException ex ) { return null ; } } @ Override public T get ( long l , TimeUnit tu ) throws InterruptedException , ExecutionException , TimeoutException { return content ; } public void set ( T c ) { notifyListeners ( ) ; content = c ; } @ Override public DummyListenableFuture < T > addListener ( GenericCompletionListener listener ) { super . addToListeners ( listener ) ; return this ; } @ Override public DummyListenableFuture < T > removeListener ( GenericCompletionListener listener ) { super . removeFromListeners ( listener ) ; return this ; } }