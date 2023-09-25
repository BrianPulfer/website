import AppLayout from '@/components/Layout/AppLayout'

export default function Layout ({ children }: { children: React.ReactNode }): JSX.Element {
  return <AppLayout>{children}</AppLayout>
}
