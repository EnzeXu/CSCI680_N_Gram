public class LiteralOpUpgrader { private final SsaMethod ssaMeth; public static void process(SsaMethod ssaMethod) { LiteralOpUpgrader dc; dc = new LiteralOpUpgrader(ssaMethod); dc.run(); } private LiteralOpUpgrader(SsaMethod ssaMethod) { this.ssaMeth = ssaMethod; } private static boolean isConstIntZeroOrKnownNull(RegisterSpec spec) { TypeBearer tb = spec.getTypeBearer(); if (tb instanceof CstLiteralBits) { CstLiteralBits clb = (CstLiteralBits) tb; return (clb.getLongBits() == 0); } return false; } private void run() { final TranslationAdvice advice = Optimizer.getAdvice(); ssaMeth.forEachInsn(new SsaInsn.Visitor() { @Override public void visitMoveInsn(NormalSsaInsn insn) { } @Override public void visitPhiInsn(PhiInsn insn) { } @Override public void visitNonMoveInsn(NormalSsaInsn insn) { Insn originalRopInsn = insn.getOriginalRopInsn(); Rop opcode = originalRopInsn.getOpcode(); RegisterSpecList sources = insn.getSources(); if (tryReplacingWithConstant(insn)) return; if (sources.size() != 2 ) { return; } if (opcode.getBranchingness() == Rop.BRANCH_IF) { if (isConstIntZeroOrKnownNull(sources.get(0))) { replacePlainInsn(insn, sources.withoutFirst(), RegOps.flippedIfOpcode(opcode.getOpcode()), null); } else if (isConstIntZeroOrKnownNull(sources.get(1))) { replacePlainInsn(insn, sources.withoutLast(), opcode.getOpcode(), null); } } else if (advice.hasConstantOperation( opcode, sources.get(0), sources.get(1))) { insn.upgradeToLiteral(); } else if (opcode.isCommutative() && advice.hasConstantOperation( opcode, sources.get(1), sources.get(0))) { insn.setNewSources( RegisterSpecList.make( sources.get(1), sources.get(0))); insn.upgradeToLiteral(); } } }); } private boolean tryReplacingWithConstant(NormalSsaInsn insn) { Insn originalRopInsn = insn.getOriginalRopInsn(); Rop opcode = originalRopInsn.getOpcode(); RegisterSpec result = insn.getResult(); if (result != null && !ssaMeth.isRegALocal(result) && opcode.getOpcode() != RegOps.CONST) { TypeBearer type = insn.getResult().getTypeBearer(); if (type.isConstant() && type.getBasicType() == Type.BT_INT) { replacePlainInsn(insn, RegisterSpecList.EMPTY, RegOps.CONST, (Constant) type); if (opcode.getOpcode() == RegOps.MOVE_RESULT_PSEUDO) { int pred = insn.getBlock().getPredecessors().nextSetBit(0); ArrayList<SsaInsn> predInsns = ssaMeth.getBlocks().get(pred).getInsns(); NormalSsaInsn sourceInsn = (NormalSsaInsn) predInsns.get(predInsns.size()-1); replacePlainInsn(sourceInsn, RegisterSpecList.EMPTY, RegOps.GOTO, null); } return true; } } return false; } private void replacePlainInsn(NormalSsaInsn insn, RegisterSpecList newSources, int newOpcode, Constant cst) { Insn originalRopInsn = insn.getOriginalRopInsn(); Rop newRop = Rops.ropFor(newOpcode, insn.getResult(), newSources, cst); Insn newRopInsn; if (cst == null) { newRopInsn = new PlainInsn(newRop, originalRopInsn.getPosition(), insn.getResult(), newSources); } else { newRopInsn = new PlainCstInsn(newRop, originalRopInsn.getPosition(), insn.getResult(), newSources, cst); } NormalSsaInsn newInsn = new NormalSsaInsn(newRopInsn, insn.getBlock()); List<SsaInsn> insns = insn.getBlock().getInsns(); ssaMeth.onInsnRemoved(insn); insns.set(insns.lastIndexOf(insn), newInsn); ssaMeth.onInsnAdded(newInsn); } }