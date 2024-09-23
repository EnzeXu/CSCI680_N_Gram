public class GetFuture < T > extends AbstractListenableFuture < T , GetCompletionListener > implements Future < T > { private final OperationFuture < Future < T > > rv ; public GetFuture ( CountDownLatch l , long opTimeout , String key , ExecutorService service ) { super ( service ) ; this . rv = new OperationFuture < Future < T > > ( key , l , opTimeout , service ) ; } public boolean cancel ( boolean ign ) { boolean result = rv . cancel ( ign ) ; notifyListeners ( ) ; return result ; } public T get ( ) throws InterruptedException , ExecutionException { Future < T > v = rv . get ( ) ; return v == null ? null : v . get ( ) ; } public T get ( long duration , TimeUnit units ) throws InterruptedException , TimeoutException , ExecutionException { Future < T > v = rv . get ( duration , units ) ; return v == null ? null : v . get ( ) ; } public OperationStatus getStatus ( ) { return rv . getStatus ( ) ; } public void set ( Future < T > d , OperationStatus s ) { rv . set ( d , s ) ; } public void setOperation ( Operation to ) { rv . setOperation ( to ) ; } public boolean isCancelled ( ) { return rv . isCancelled ( ) ; } public boolean isDone ( ) { return rv . isDone ( ) ; } @ Override public GetFuture < T > addListener ( GetCompletionListener listener ) { super . addToListeners ( ( GenericCompletionListener ) listener ) ; return this ; } @ Override public GetFuture < T > removeListener ( GetCompletionListener listener ) { super . removeFromListeners ( ( GenericCompletionListener ) listener ) ; return this ; } public void signalComplete ( ) { notifyListeners ( ) ; } }