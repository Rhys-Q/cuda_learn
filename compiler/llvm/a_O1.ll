; ModuleID = 'a.cc'
source_filename = "a.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%class.Animal = type { i32 (...)** }
%class.Cat = type { %class.Animal, i32, i8* }
%class.Lion = type { %class.Animal, i32, i8* }

$_ZN3Cat5printEv = comdat any

$_ZN4Lion5printEv = comdat any

$_ZTV3Cat = comdat any

$_ZTS3Cat = comdat any

$_ZTS6Animal = comdat any

$_ZTI6Animal = comdat any

$_ZTI3Cat = comdat any

$_ZZN3Cat5printEvE3cnt = comdat any

$_ZTV4Lion = comdat any

$_ZTS4Lion = comdat any

$_ZTI4Lion = comdat any

@.str = private unnamed_addr constant [25 x i8] c"Dog age = %d, name = %s\0A\00", align 1
@.str.1 = private unnamed_addr constant [4 x i8] c"dog\00", align 1
@.str.2 = private unnamed_addr constant [4 x i8] c"cat\00", align 1
@.str.3 = private unnamed_addr constant [5 x i8] c"lion\00", align 1
@.str.4 = private unnamed_addr constant [44 x i8] c"\E8\AF\B7\E8\BE\93\E5\85\A5\E5\8A\A8\E7\89\A9\E7\B1\BB\E5\9E\8B\EF\BC\881: \E7\8B\97, 0: \E7\8C\AB\EF\BC\89: \00", align 1
@.str.5 = private unnamed_addr constant [3 x i8] c"%d\00", align 1
@.str.6 = private unnamed_addr constant [18 x i8] c"\E6\89\93\E5\8D\B0\E7\BB\93\E6\9E\9C: %d\0A\00", align 1
@_ZTV3Cat = linkonce_odr dso_local unnamed_addr constant { [3 x i8*] } { [3 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTI3Cat to i8*), i8* bitcast (void (%class.Cat*)* @_ZN3Cat5printEv to i8*)] }, comdat, align 8
@_ZTVN10__cxxabiv120__si_class_type_infoE = external global i8*
@_ZTS3Cat = linkonce_odr dso_local constant [5 x i8] c"3Cat\00", comdat, align 1
@_ZTVN10__cxxabiv117__class_type_infoE = external global i8*
@_ZTS6Animal = linkonce_odr dso_local constant [8 x i8] c"6Animal\00", comdat, align 1
@_ZTI6Animal = linkonce_odr dso_local constant { i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv117__class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @_ZTS6Animal, i32 0, i32 0) }, comdat, align 8
@_ZTI3Cat = linkonce_odr dso_local constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([5 x i8], [5 x i8]* @_ZTS3Cat, i32 0, i32 0), i8* bitcast ({ i8*, i8* }* @_ZTI6Animal to i8*) }, comdat, align 8
@_ZZN3Cat5printEvE3cnt = linkonce_odr dso_local local_unnamed_addr global i32 0, comdat, align 4
@.str.7 = private unnamed_addr constant [35 x i8] c"Cat age = %d, name = %s, cnt = %d\0A\00", align 1
@_ZTV4Lion = linkonce_odr dso_local unnamed_addr constant { [3 x i8*] } { [3 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTI4Lion to i8*), i8* bitcast (void (%class.Lion*)* @_ZN4Lion5printEv to i8*)] }, comdat, align 8
@_ZTS4Lion = linkonce_odr dso_local constant [6 x i8] c"4Lion\00", comdat, align 1
@_ZTI4Lion = linkonce_odr dso_local constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @_ZTS4Lion, i32 0, i32 0), i8* bitcast ({ i8*, i8* }* @_ZTI6Animal to i8*) }, comdat, align 8
@.str.8 = private unnamed_addr constant [26 x i8] c"Lion age = %d, name = %s\0A\00", align 1
@.str.9 = private unnamed_addr constant [7 x i8] c"global\00", align 1

; Function Attrs: mustprogress nofree nounwind uwtable
define dso_local void @_Z8PrintDog3Dog(i32 %0, i8* %1) local_unnamed_addr #0 {
  %3 = call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([25 x i8], [25 x i8]* @.str, i64 0, i64 0), i32 noundef %0, i8* noundef %1)
  ret void
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(i8* nocapture noundef readonly, ...) local_unnamed_addr #1

; Function Attrs: mustprogress uwtable
define dso_local void @_Z11PrintAnimalP6Animal(%class.Animal* noundef %0) local_unnamed_addr #2 {
  %2 = bitcast %class.Animal* %0 to void (%class.Animal*)***
  %3 = load void (%class.Animal*)**, void (%class.Animal*)*** %2, align 8, !tbaa !5
  %4 = load void (%class.Animal*)*, void (%class.Animal*)** %3, align 8
  call void %4(%class.Animal* noundef nonnull align 8 dereferenceable(8) %0)
  ret void
}

; Function Attrs: uwtable
define dso_local noundef i32 @_Z11PrintAnimalb(i1 noundef zeroext %0) local_unnamed_addr #3 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  br i1 %0, label %2, label %7

2:                                                ; preds = %1, %2
  %3 = phi i32 [ %5, %2 ], [ 0, %1 ]
  %4 = call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([25 x i8], [25 x i8]* @.str, i64 0, i64 0), i32 noundef 1, i8* noundef getelementptr inbounds ([4 x i8], [4 x i8]* @.str.1, i64 0, i64 0)) #8
  %5 = add nuw nsw i32 %3, 1
  %6 = icmp eq i32 %5, 3
  br i1 %6, label %31, label %2, !llvm.loop !8

7:                                                ; preds = %1
  %8 = call noalias noundef nonnull dereferenceable(24) i8* @_Znwm(i64 noundef 24) #9
  %9 = bitcast i8* %8 to %class.Cat*
  %10 = getelementptr inbounds %class.Cat, %class.Cat* %9, i64 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTV3Cat, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %10, align 8, !tbaa !5
  %11 = getelementptr inbounds %class.Cat, %class.Cat* %9, i64 0, i32 1
  store i32 1, i32* %11, align 8, !tbaa !11
  %12 = getelementptr inbounds %class.Cat, %class.Cat* %9, i64 0, i32 2
  store i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.2, i64 0, i64 0), i8** %12, align 8, !tbaa !16
  %13 = call noalias noundef nonnull dereferenceable(24) i8* @_Znwm(i64 noundef 24) #9
  %14 = bitcast i8* %13 to %class.Lion*
  %15 = getelementptr inbounds %class.Lion, %class.Lion* %14, i64 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTV4Lion, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %15, align 8, !tbaa !5
  %16 = getelementptr inbounds %class.Lion, %class.Lion* %14, i64 0, i32 1
  store i32 1, i32* %16, align 8, !tbaa !17
  %17 = getelementptr inbounds %class.Lion, %class.Lion* %14, i64 0, i32 2
  store i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.3, i64 0, i64 0), i8** %17, align 8, !tbaa !19
  %18 = getelementptr %class.Cat, %class.Cat* %9, i64 0, i32 0
  %19 = bitcast i8* %8 to void (%class.Animal*)***
  %20 = getelementptr %class.Lion, %class.Lion* %14, i64 0, i32 0
  %21 = bitcast i8* %13 to void (%class.Animal*)***
  br label %22

22:                                               ; preds = %7, %22
  %23 = phi i32 [ 0, %7 ], [ %28, %22 ]
  %24 = load void (%class.Animal*)**, void (%class.Animal*)*** %19, align 8, !tbaa !5
  %25 = load void (%class.Animal*)*, void (%class.Animal*)** %24, align 8
  call void %25(%class.Animal* noundef nonnull align 8 dereferenceable(8) %18)
  %26 = load void (%class.Animal*)**, void (%class.Animal*)*** %21, align 8, !tbaa !5
  %27 = load void (%class.Animal*)*, void (%class.Animal*)** %26, align 8
  call void %27(%class.Animal* noundef nonnull align 8 dereferenceable(8) %20)
  %28 = add nuw nsw i32 %23, 1
  %29 = icmp eq i32 %28, 3
  br i1 %29, label %30, label %22, !llvm.loop !20

30:                                               ; preds = %22
  call void @_ZdlPv(i8* noundef %8) #10
  call void @_ZdlPv(i8* noundef %13) #10
  br label %31

31:                                               ; preds = %2, %30
  %32 = phi i32 [ 1, %30 ], [ 2, %2 ]
  ret i32 %32
}

; Function Attrs: argmemonly mustprogress nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #4

; Function Attrs: argmemonly mustprogress nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #4

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull i8* @_Znwm(i64 noundef) local_unnamed_addr #5

declare i32 @__gxx_personality_v0(...)

; Function Attrs: nobuiltin nounwind
declare void @_ZdlPv(i8* noundef) local_unnamed_addr #6

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main() local_unnamed_addr #7 {
  %1 = alloca i8, align 1
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %1) #8
  %2 = call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([44 x i8], [44 x i8]* @.str.4, i64 0, i64 0))
  %3 = call i32 (i8*, ...) @__isoc99_scanf(i8* noundef getelementptr inbounds ([3 x i8], [3 x i8]* @.str.5, i64 0, i64 0), i8* noundef nonnull %1)
  %4 = load i8, i8* %1, align 1, !tbaa !21, !range !23
  %5 = icmp ne i8 %4, 0
  %6 = call noundef i32 @_Z11PrintAnimalb(i1 noundef zeroext %5), !range !24
  %7 = call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([18 x i8], [18 x i8]* @.str.6, i64 0, i64 0), i32 noundef %6)
  %8 = call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([25 x i8], [25 x i8]* @.str, i64 0, i64 0), i32 noundef 2, i8* noundef getelementptr inbounds ([7 x i8], [7 x i8]* @.str.9, i64 0, i64 0)) #8
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %1) #8
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @__isoc99_scanf(i8* nocapture noundef readonly, ...) local_unnamed_addr #1

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZN3Cat5printEv(%class.Cat* noundef nonnull align 8 dereferenceable(24) %0) unnamed_addr #2 comdat align 2 {
  %2 = load i32, i32* @_ZZN3Cat5printEvE3cnt, align 4, !tbaa !25
  %3 = add nsw i32 %2, 1
  store i32 %3, i32* @_ZZN3Cat5printEvE3cnt, align 4, !tbaa !25
  %4 = getelementptr inbounds %class.Cat, %class.Cat* %0, i64 0, i32 1
  %5 = load i32, i32* %4, align 8, !tbaa !11
  %6 = getelementptr inbounds %class.Cat, %class.Cat* %0, i64 0, i32 2
  %7 = load i8*, i8** %6, align 8, !tbaa !16
  %8 = call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([35 x i8], [35 x i8]* @.str.7, i64 0, i64 0), i32 noundef %5, i8* noundef %7, i32 noundef %3)
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZN4Lion5printEv(%class.Lion* noundef nonnull align 8 dereferenceable(24) %0) unnamed_addr #2 comdat align 2 {
  %2 = getelementptr inbounds %class.Lion, %class.Lion* %0, i64 0, i32 1
  %3 = load i32, i32* %2, align 8, !tbaa !17
  %4 = getelementptr inbounds %class.Lion, %class.Lion* %0, i64 0, i32 2
  %5 = load i8*, i8** %4, align 8, !tbaa !19
  %6 = call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([26 x i8], [26 x i8]* @.str.8, i64 0, i64 0), i32 noundef %3, i8* noundef %5)
  ret void
}

attributes #0 = { mustprogress nofree nounwind uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nofree nounwind "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { mustprogress uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #4 = { argmemonly mustprogress nofree nosync nounwind willreturn }
attributes #5 = { nobuiltin allocsize(0) "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #6 = { nobuiltin nounwind "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #7 = { mustprogress norecurse uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #8 = { nounwind }
attributes #9 = { builtin allocsize(0) }
attributes #10 = { builtin nounwind }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 1}
!4 = !{!"Ubuntu clang version 14.0.0-1ubuntu1.1"}
!5 = !{!6, !6, i64 0}
!6 = !{!"vtable pointer", !7, i64 0}
!7 = !{!"Simple C++ TBAA"}
!8 = distinct !{!8, !9, !10}
!9 = !{!"llvm.loop.mustprogress"}
!10 = !{!"llvm.loop.unroll.disable"}
!11 = !{!12, !13, i64 8}
!12 = !{!"_ZTS3Cat", !13, i64 8, !15, i64 16}
!13 = !{!"int", !14, i64 0}
!14 = !{!"omnipotent char", !7, i64 0}
!15 = !{!"any pointer", !14, i64 0}
!16 = !{!12, !15, i64 16}
!17 = !{!18, !13, i64 8}
!18 = !{!"_ZTS4Lion", !13, i64 8, !15, i64 16}
!19 = !{!18, !15, i64 16}
!20 = distinct !{!20, !9, !10}
!21 = !{!22, !22, i64 0}
!22 = !{!"bool", !14, i64 0}
!23 = !{i8 0, i8 2}
!24 = !{i32 1, i32 3}
!25 = !{!13, !13, i64 0}
