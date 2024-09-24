public class OperationFuture<T> extends AbstractListenableFuture<T, OperationCompletionListener> implements Future<T> { private final CountDownLatch latch; private final AtomicReference<T> objRef; protected OperationStatus status; private final long timeout; private Operation op; private final String key; private Long cas; public OperationFuture(String k, CountDownLatch l, long opTimeout, ExecutorService service) { this(k, l, new AtomicReference<T>(null), opTimeout, service); } public OperationFuture(String k, CountDownLatch l, AtomicReference<T> oref, long opTimeout, ExecutorService service) { super(service); latch = l; objRef = oref; status = null; timeout = opTimeout; key = k; cas = null; } public boolean cancel(boolean ign) { assert op != null : "No operation"; op.cancel(); notifyListeners(); return op.getState() == OperationState.WRITE_QUEUED; } public boolean cancel() { assert op != null : "No operation"; op.cancel(); notifyListeners(); return op.getState() == OperationState.WRITE_QUEUED; } public T get() throws InterruptedException, ExecutionException { try { return get(timeout, TimeUnit.MILLISECONDS); } catch (TimeoutException e) { throw new RuntimeException("Timed out waiting for operation", e); } } public T get(long duration, TimeUnit units) throws InterruptedException, TimeoutException, ExecutionException { if (!latch.await(duration, units)) { MemcachedConnection.opTimedOut(op); if (op != null) { op.timeOut(); } throw new CheckedOperationTimeoutException( "Timed out waiting for operation", op); } else { MemcachedConnection.opSucceeded(op); } if (op != null && op.hasErrored()) { throw new ExecutionException(op.getException()); } if (isCancelled()) { throw new ExecutionException(new CancellationException("Cancelled")); } if (op != null && op.isTimedOut()) { throw new ExecutionException(new CheckedOperationTimeoutException( "Operation timed out.", op)); } return objRef.get(); } public String getKey() { return key; } public void setCas(long inCas) { this.cas = inCas; } public Long getCas() { if (cas == null) { try { get(); } catch (InterruptedException e) { status = new OperationStatus(false, "Interrupted", StatusCode.INTERRUPTED); } catch (ExecutionException e) { getLogger().warn("Error getting cas of operation", e); } } if (cas == null && status.isSuccess()) { throw new UnsupportedOperationException("This operation doesn't return" + "a cas value."); } return cas; } public OperationStatus getStatus() { if (status == null) { try { get(); } catch (InterruptedException e) { status = new OperationStatus(false, "Interrupted", StatusCode.INTERRUPTED); } catch (ExecutionException e) { getLogger().warn("Error getting status of operation", e); } } return status; } public void set(T o, OperationStatus s) { objRef.set(o); status = s; } public void setOperation(Operation to) { op = to; } public boolean isCancelled() { assert op != null : "No operation"; return op.isCancelled(); } public boolean isDone() { assert op != null : "No operation"; return latch.getCount() == 0 || op.isCancelled() || op.getState() == OperationState.COMPLETE; } @Override public OperationFuture<T> addListener(OperationCompletionListener listener) { super.addToListeners((GenericCompletionListener) listener); return this; } @Override public OperationFuture<T> removeListener( OperationCompletionListener listener) { super.removeFromListeners((GenericCompletionListener) listener); return this; } public void signalComplete() { notifyListeners(); } }