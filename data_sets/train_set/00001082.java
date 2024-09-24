public class BootstrapMethodArgumentsList extends FixedSizeList { public BootstrapMethodArgumentsList(int count) { super(count); } public Constant get(int n) { return (Constant) get0(n); } public void set(int n, Constant cst) { if (cst instanceof CstString || cst instanceof CstType || cst instanceof CstInteger || cst instanceof CstLong || cst instanceof CstFloat || cst instanceof CstDouble || cst instanceof CstMethodHandle || cst instanceof CstProtoRef) { set0(n, cst); } else { Class<?> klass = cst.getClass(); throw new IllegalArgumentException("bad type for bootstrap argument: " + klass); } } }